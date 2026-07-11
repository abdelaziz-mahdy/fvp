/*
 * Copyright (c) 2026 WangBin <wbsecg1 at gmail.com>
 */
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Cross-context RGBA_1010102 sampling probe.
//
// Some drivers (e.g. PowerVR BXE on Realtek TV SoCs, wang-bin/fvp#374) accept
// a 10-bit EGLConfig, allocate and render into RGBA_1010102 buffers without
// any EGL/GL error — but cannot sample those buffers consistently from a
// *different* GL context (IMGSRV "IsTextureConsistent" failures), which
// corrupts every frame the Flutter engine consumes. Since no error surfaces
// through the API, the only reliable detection is to reproduce the handoff:
//
//   1. create an AImageReader with AIMAGE_FORMAT_PRIVATE (producer-defined
//      format) and render a known pattern into its window from an EGL window
//      surface using a 10-bit config — the driver then allocates its native
//      (vendor tiled/compressed) RGBA_1010102 window buffers, exactly like
//      the real video path. A directly allocated AHardwareBuffer is NOT
//      enough: it skips the EGL window allocation path and its vendor
//      compression, and round-trips fine on drivers that corrupt real
//      window buffers.
//   2. acquire the produced buffer and import it as an EGLImage in a second,
//      unshared context (what the Flutter engine does), sample as an
//      external texture and read back
//
// A mismatch decides to force GLRenderAPI.depth = 8 before
// updateNativeSurface(). Any probe-infrastructure failure returns "ok" so
// healthy devices are never punished. Probed once per process.
//
// Env overrides for testing: FVP_RGB10A2_PROBE=0 (skip, assume ok),
// FVP_RGB10A2_PROBE=force8 (skip, assume broken).

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>
#include <android/native_window.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

using std::clog;
using std::endl;

namespace {

// libmediandk symbols are resolved at runtime: AImageReader_newWithUsage and
// AImage_getHardwareBuffer are API 26+, and direct linking would break plugin
// load on older devices (which won't see 1010102 anyway).
struct AImageReader;
struct AImage;
typedef int (*AImageReader_newWithUsage_t)(int32_t, int32_t, int32_t, uint64_t, int32_t, AImageReader**);
typedef int (*AImageReader_getWindow_t)(AImageReader*, ANativeWindow**);
typedef int (*AImageReader_acquireNextImage_t)(AImageReader*, AImage**);
typedef void (*AImageReader_delete_t)(AImageReader*);
typedef int (*AImage_getHardwareBuffer_t)(const AImage*, AHardwareBuffer**);
typedef void (*AImage_delete_t)(AImage*);

constexpr int32_t kFormatPrivate = 0x22; // AIMAGE_FORMAT_PRIVATE: producer-defined

constexpr int kSrcSize = 512; // large enough for vendor tiled/compressed layouts
constexpr int kBlock = 8;     // 8 px * 4 Bpp = 32 bytes: the observed corruption burst size
constexpr int kBlocks = kSrcSize / kBlock; // 64x64 blocks == dst pixels
constexpr int kDstSize = kBlocks;

// Vendor framebuffer compression decodes FLAT content correctly even on
// broken drivers (the real corruption spares flat areas and hits detail).
// So the pattern must be high-frequency:每 8px block gets a pseudo-random
// color from a hash, quantized to multiples of 85 for lossless 10->8 bit
// round-trips.
void blockColor(int bx, int by, uint8_t* rgb) {
  const uint32_t v = (uint32_t)(bx * 73856093u) ^ (uint32_t)(by * 19349663u) ^ 0x9E3779B9u;
  rgb[0] = (uint8_t)(((v >> 0) & 3) * 85);
  rgb[1] = (uint8_t)(((v >> 8) & 3) * 85);
  rgb[2] = (uint8_t)(((v >> 16) & 3) * 85);
}

bool matches(const uint8_t* px, const uint8_t* rgb) {
  return abs(int(px[0]) - rgb[0]) <= 24 && abs(int(px[1]) - rgb[1]) <= 24 &&
         abs(int(px[2]) - rgb[2]) <= 24;
}

struct ProbeCleanup {
  void* ndk = nullptr;
  EGLDisplay dpy = EGL_NO_DISPLAY;
  EGLSurface winSurf = EGL_NO_SURFACE;
  EGLSurface pbufB = EGL_NO_SURFACE;
  EGLContext ctxA = EGL_NO_CONTEXT;
  EGLContext ctxB = EGL_NO_CONTEXT;
  EGLImageKHR image = EGL_NO_IMAGE_KHR;
  AImageReader* reader = nullptr;
  AImage* img = nullptr;
  AImageReader_delete_t readerDelete = nullptr;
  AImage_delete_t imageDelete = nullptr;
  PFNEGLDESTROYIMAGEKHRPROC destroyImage = nullptr;
  // Whatever was current on this thread before the probe ran — restored on
  // exit so callers with a live EGL context are not clobbered.
  EGLDisplay oldDpy = EGL_NO_DISPLAY;
  EGLContext oldCtx = EGL_NO_CONTEXT;
  EGLSurface oldDraw = EGL_NO_SURFACE;
  EGLSurface oldRead = EGL_NO_SURFACE;

  void saveCurrent() {
    oldDpy = eglGetCurrentDisplay();
    oldCtx = eglGetCurrentContext();
    oldDraw = eglGetCurrentSurface(EGL_DRAW);
    oldRead = eglGetCurrentSurface(EGL_READ);
  }

  ~ProbeCleanup() {
    if (dpy != EGL_NO_DISPLAY) {
      if (oldCtx != EGL_NO_CONTEXT && oldDpy != EGL_NO_DISPLAY) {
        eglMakeCurrent(oldDpy, oldDraw, oldRead, oldCtx);
      } else {
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      }
      if (image != EGL_NO_IMAGE_KHR && destroyImage) destroyImage(dpy, image);
      if (winSurf != EGL_NO_SURFACE) eglDestroySurface(dpy, winSurf);
      if (pbufB != EGL_NO_SURFACE) eglDestroySurface(dpy, pbufB);
      if (ctxA != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctxA);
      if (ctxB != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctxB);
    }
    if (img && imageDelete) imageDelete(img);
    if (reader && readerDelete) readerDelete(reader);
    // Last: the deleters above live in this library.
    if (ndk) dlclose(ndk);
  }
};

// true = 10-bit path verified broken. Everything else (including probe
// infrastructure failures, each logged with its step) = false.
bool probeShowsBroken() {
  void* ndk = dlopen("libmediandk.so", RTLD_NOW | RTLD_LOCAL);
  if (!ndk) {
    clog << "rgb10a2 probe skip: no libmediandk" << endl;
    return false;
  }
  auto newWithUsage = (AImageReader_newWithUsage_t)dlsym(ndk, "AImageReader_newWithUsage");
  auto getWindow = (AImageReader_getWindow_t)dlsym(ndk, "AImageReader_getWindow");
  auto acquireNext = (AImageReader_acquireNextImage_t)dlsym(ndk, "AImageReader_acquireNextImage");
  auto readerDelete = (AImageReader_delete_t)dlsym(ndk, "AImageReader_delete");
  auto getHwBuffer = (AImage_getHardwareBuffer_t)dlsym(ndk, "AImage_getHardwareBuffer");
  auto imageDelete = (AImage_delete_t)dlsym(ndk, "AImage_delete");
  if (!newWithUsage || !getWindow || !acquireNext || !readerDelete || !getHwBuffer || !imageDelete) {
    clog << "rgb10a2 probe skip: mediandk symbols missing (pre-26?)" << endl;
    dlclose(ndk);
    return false;
  }
  auto getNativeClientBuffer =
      (PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC)eglGetProcAddress("eglGetNativeClientBufferANDROID");
  auto createImage = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
  auto destroyImage = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
  auto imageTargetTexture =
      (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
  if (!getNativeClientBuffer || !createImage || !destroyImage || !imageTargetTexture) {
    clog << "rgb10a2 probe skip: EGL extension functions missing" << endl;
    return false;
  }

  ProbeCleanup c;
  c.ndk = ndk;
  c.saveCurrent();
  c.destroyImage = destroyImage;
  c.readerDelete = readerDelete;
  c.imageDelete = imageDelete;

  c.dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  EGLint maj, min;
  if (c.dpy == EGL_NO_DISPLAY || !eglInitialize(c.dpy, &maj, &min)) {
    c.dpy = EGL_NO_DISPLAY;
    clog << "rgb10a2 probe skip: no EGL display" << endl;
    return false;
  }

  // Only meaningful when a 10-bit config exists for MDK to pick.
  const EGLint attrs10[] = {EGL_RED_SIZE, 10, EGL_GREEN_SIZE, 10, EGL_BLUE_SIZE, 10,
                            EGL_ALPHA_SIZE, 2, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                            EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_NONE};
  EGLConfig cfg10;
  EGLint n = 0;
  if (!eglChooseConfig(c.dpy, attrs10, &cfg10, 1, &n) || n < 1) {
    clog << "rgb10a2 probe skip: no 10-bit EGLConfig" << endl;
    return false;
  }

  // BufferQueue consumer with a producer-defined format: the EGL window
  // surface below (10-bit config) makes the driver allocate its native
  // compressed RGBA_1010102 window buffers — the exact real-path allocation.
  const uint64_t usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
  const int rrc = newWithUsage(kSrcSize, kSrcSize, kFormatPrivate, usage, 2, &c.reader);
  if (rrc != 0 || !c.reader) {
    clog << "rgb10a2 probe skip: AImageReader(PRIVATE) failed: " << rrc << endl;
    return false;
  }
  ANativeWindow* window = nullptr; // owned by the reader
  if (getWindow(c.reader, &window) != 0 || !window) {
    clog << "rgb10a2 probe skip: no reader window" << endl;
    return false;
  }

  const EGLint ctxAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  const EGLint pbAttrs[] = {EGL_WIDTH, kDstSize, EGL_HEIGHT, kDstSize, EGL_NONE};

  // --- Context A (producer): 10-bit window surface, like MDK's renderer. ---
  c.ctxA = eglCreateContext(c.dpy, cfg10, EGL_NO_CONTEXT, ctxAttrs);
  if (c.ctxA == EGL_NO_CONTEXT) {
    clog << "rgb10a2 probe skip: ctxA create failed" << endl;
    return false;
  }
  c.winSurf = eglCreateWindowSurface(c.dpy, cfg10, window, nullptr);
  if (c.winSurf == EGL_NO_SURFACE) {
    clog << "rgb10a2 probe skip: window surface failed: 0x" << std::hex << eglGetError()
         << std::dec << endl;
    return false;
  }
  if (!eglMakeCurrent(c.dpy, c.winSurf, c.winSurf, c.ctxA)) {
    clog << "rgb10a2 probe skip: makeCurrent A failed" << endl;
    return false;
  }
  glEnable(GL_SCISSOR_TEST);
  for (int by = 0; by < kBlocks; by++) {
    for (int bx = 0; bx < kBlocks; bx++) {
      uint8_t rgb[3];
      blockColor(bx, by, rgb);
      glScissor(bx * kBlock, by * kBlock, kBlock, kBlock);
      glClearColor(rgb[0] / 255.f, rgb[1] / 255.f, rgb[2] / 255.f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT);
    }
  }
  glDisable(GL_SCISSOR_TEST);
  glFinish();
  if (!eglSwapBuffers(c.dpy, c.winSurf)) {
    clog << "rgb10a2 probe skip: swapBuffers failed" << endl;
    return false;
  }
  eglMakeCurrent(c.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

  // The queued buffer lands in the reader asynchronously.
  for (int tries = 0; tries < 50 && !c.img; tries++) {
    if (acquireNext(c.reader, &c.img) != 0) c.img = nullptr;
    if (!c.img) usleep(2000);
  }
  if (!c.img) {
    clog << "rgb10a2 probe skip: no image acquired" << endl;
    return false;
  }
  AHardwareBuffer* ahb = nullptr; // owned by the AImage
  if (getHwBuffer(c.img, &ahb) != 0 || !ahb) {
    clog << "rgb10a2 probe skip: no AHardwareBuffer" << endl;
    return false;
  }

  EGLClientBuffer clientBuf = getNativeClientBuffer(ahb);
  if (!clientBuf) {
    clog << "rgb10a2 probe skip: no client buffer" << endl;
    return false;
  }
  c.image = createImage(c.dpy, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, clientBuf, nullptr);
  if (c.image == EGL_NO_IMAGE_KHR) {
    clog << "rgb10a2 probe skip: eglCreateImageKHR failed: 0x" << std::hex << eglGetError()
         << std::dec << endl;
    return false;
  }

  const EGLint attrs8[] = {EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                           EGL_ALPHA_SIZE, 8, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                           EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
  EGLConfig cfg8;
  if (!eglChooseConfig(c.dpy, attrs8, &cfg8, 1, &n) || n < 1) {
    clog << "rgb10a2 probe skip: no 8-bit pbuffer config" << endl;
    return false;
  }

  // --- Context B (consumer): unshared, like the Flutter engine's. ---
  c.ctxB = eglCreateContext(c.dpy, cfg8, EGL_NO_CONTEXT, ctxAttrs);
  if (c.ctxB == EGL_NO_CONTEXT) {
    clog << "rgb10a2 probe skip: ctxB create failed" << endl;
    return false;
  }
  c.pbufB = eglCreatePbufferSurface(c.dpy, cfg8, pbAttrs);
  if (c.pbufB == EGL_NO_SURFACE) {
    clog << "rgb10a2 probe skip: pbufB failed" << endl;
    return false;
  }
  if (!eglMakeCurrent(c.dpy, c.pbufB, c.pbufB, c.ctxB)) {
    clog << "rgb10a2 probe skip: makeCurrent B failed" << endl;
    return false;
  }

  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_EXTERNAL_OES, tex);
  imageTargetTexture(GL_TEXTURE_EXTERNAL_OES, c.image);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  static const char* kVs =
      "attribute vec2 a;varying vec2 v;"
      "void main(){v=a*0.5+0.5;gl_Position=vec4(a,0.0,1.0);}";
  static const char* kFs =
      "#extension GL_OES_EGL_image_external : require\n"
      "precision mediump float;uniform samplerExternalOES t;varying vec2 v;"
      "void main(){gl_FragColor=texture2D(t,v);}";
  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &kVs, nullptr);
  glCompileShader(vs);
  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &kFs, nullptr);
  glCompileShader(fs);
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glBindAttribLocation(prog, 0, "a");
  glLinkProgram(prog);
  GLint linked = 0;
  glGetProgramiv(prog, GL_LINK_STATUS, &linked);
  if (!linked) {
    clog << "rgb10a2 probe skip: shader link failed" << endl;
    return false;
  }
  glUseProgram(prog);
  glUniform1i(glGetUniformLocation(prog, "t"), 0);
  static const GLfloat verts[] = {-1, -1, 1, -1, -1, 1, 1, 1};
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, verts);
  glEnableVertexAttribArray(0);
  glViewport(0, 0, kDstSize, kDstSize);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glFinish();

  // Each dst pixel maps NEAREST onto exactly one 8px source block. Compare
  // every pixel against the hash color for both Y orientations (the flip is
  // driver-dependent); corruption from a mis-decoded compressed buffer
  // matches neither. A small allowance covers stray filtering artifacts.
  static uint8_t out[kDstSize * kDstSize * 4];
  glReadPixels(0, 0, kDstSize, kDstSize, GL_RGBA, GL_UNSIGNED_BYTE, out);
  const GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    clog << "rgb10a2 probe skip: GL error 0x" << std::hex << e << std::dec << endl;
    return false;
  }
  int badUp = 0, badDown = 0;
  for (int y = 0; y < kDstSize; y++) {
    for (int x = 0; x < kDstSize; x++) {
      const uint8_t* px = &out[(y * kDstSize + x) * 4];
      uint8_t up[3], down[3];
      blockColor(x, y, up);
      blockColor(x, kBlocks - 1 - y, down);
      if (!matches(px, up)) badUp++;
      if (!matches(px, down)) badDown++;
    }
  }
  const int bad = badUp < badDown ? badUp : badDown;
  clog << "rgb10a2 cross-context probe: " << bad << "/" << kDstSize * kDstSize
       << " samples corrupt" << endl;
  return bad > kDstSize; // > ~1.5% corrupt = broken
}

} // namespace

// True when RGBA_1010102 buffers survive cross-context sampling on this
// driver (or when the probe cannot run). Probed once; magic-static
// initialization makes concurrent first calls safe.
bool fvpRgb10a2CrossContextOk() {
  static const bool ok = [] {
    if (const char* env = getenv("FVP_RGB10A2_PROBE")) {
      if (!strcmp(env, "0")) return true;
      if (!strcmp(env, "force8")) return false;
    }
    const bool broken = probeShowsBroken();
    clog << "rgb10a2 cross-context sampling ok: " << !broken << endl;
    return !broken;
  }();
  return ok;
}
