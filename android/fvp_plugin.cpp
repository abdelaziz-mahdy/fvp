/*
 * Copyright (c) 2023-2026 WangBin <wbsecg1 at gmail.com>
 */
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <jni.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <EGL/egl.h>
//#include <vulkan/vulkan.h> // before any mdk header
#include <mdk/Player.h>
#include <mdk/MediaInfo.h>
#include <mdk/RenderAPI.h>
#include <cassert>
#include <cstdlib>
#include <unordered_map>
#include <iostream>
#include <sys/system_properties.h>

using namespace std;

class TexturePlayer final : public mdk::Player
{
public:
    TexturePlayer(jlong handle)
        : mdk::Player(reinterpret_cast<mdkPlayerAPI*>(handle))
    {
    }

    int width = 0;
    int height = 0;
    jobject surface = nullptr;
    void* vo_opaque = nullptr; // can change by TextureRegistry.SurfaceProducer.Callback
    bool directSurface = false; // decoder renders straight to the surface (FVP_DIRECT_SURFACE)
    ANativeWindow* window = nullptr; // held for tunnel mode; released on detach
private:
};

static unordered_map<int64_t, shared_ptr<TexturePlayer>> players;

// Some drivers expose RGBA_1010102 window configs but mark them all
// EGL_NON_CONFORMANT_CONFIG (e.g. PowerVR BXE on Realtek TVs): rendering into
// one from a shared context corrupts the output (#374). mdk prefers a 10bit
// EGLConfig for its render target; if the driver has no conformant 10bit
// window config, request an 8bit target instead. Returns true when the
// fallback is needed. Probed once, EGL only (no context/surface created).
static bool no10BitConformantWindowConfig()
{
    static const bool no10bit = []{
        EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (dpy == EGL_NO_DISPLAY)
            return false;
        EGLint major = 0, minor = 0;
        if (!eglInitialize(dpy, &major, &minor)) // ref counted, no eglTerminate
            return false;
        const EGLint attrs[] = {
            EGL_CONFIG_CAVEAT, EGL_NONE, // exclude slow/non-conformant configs
            EGL_BUFFER_SIZE, 32,
            EGL_RED_SIZE, 10,
            EGL_GREEN_SIZE, 10,
            EGL_BLUE_SIZE, 10,
            EGL_ALPHA_SIZE, 2,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT | 0x40/*EGL_OPENGL_ES3_BIT*/,
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_NONE
        };
        EGLint count = 0;
        const EGLBoolean ok = eglChooseConfig(dpy, attrs, nullptr, 0, &count);
        clog << "conformant rgb10a2 window EGLConfigs: " << count << endl;
        return ok && count <= 0;
    }();
    return no10bit;
}


extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    clog << "JNI_OnLoad" << endl;
    mdk::javaVM(vm);
    mdk::SetGlobalOption("profiler.gpu", 1);

    JNIEnv *env = nullptr;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK || !env) {
        clog << "GetEnv for JNI_VERSION_1_4 failed" << endl;
        return -1;
    }

    return JNI_VERSION_1_4;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    clog << "JNI_OnUnload" << endl;
}
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mediadevkit_fvp_FvpPlugin_nativeSetSurface(JNIEnv *env, jobject thiz, jlong player_handle,
                                                    jlong tex_id, jobject surface, jint w, jint h, jboolean tunnel) {
    if (!player_handle || !surface) {
        if (auto it = players.find(tex_id); it != players.end()) {
            auto& player = it->second;
            auto s = player->surface;
            auto win = player->window;
            if (player->directSurface) {
                // The codec is rendering into the surface that is being
                // destroyed right now. Detach it BEFORE the surface dies: a
                // MediaCodec left bound to a dead surface wedges (dequeue
                // -10000, "can not return buffer to native window") and MDK
                // marks the stream as decode-error — the next surface then
                // never shows a frame. The re-attach re-opens the decoder with
                // the new surface.
                //
                // An empty decoder list releases the codec without opening a
                // replacement. Re-opening one here ({"AMediaCodec"}) costs
                // ~600 ms on the calling thread — and this runs on the Android
                // UI thread inside SurfaceHolder.Callback.surfaceDestroyed —
                // because MDK creates+configures+starts a fresh codec and
                // re-primes the pipeline (measured: 37 skipped frames, the
                // whole exit animation frozen, on a TCL/Realtek TV with a 30 s
                // buffer). Nothing can be displayed without a surface anyway.
                player->setDecoders(mdk::MediaType::Video, {});
            }
            player->updateNativeSurface(nullptr);
            players.erase(it);
            if (win) {
                ANativeWindow_release(win);
            }
            if (s) {
                env->DeleteGlobalRef(s);
            }
        } else {
            clog << "player not found(already removed?) for textureId " + std::to_string(tex_id) + " surface " + std::to_string((intptr_t)surface) << endl;
        }
        return;
    }
    assert(surface && "null surface");
    auto player = make_shared<TexturePlayer>(player_handle);
    clog << __func__ << endl;
    if (tunnel) { // TODO: tunel via ffi + global var
        player->surface = env->NewGlobalRef(surface);
        player->setProperty("video.decoder", "surface=" + std::to_string((intptr_t)player->surface));
        // Platform-view surfaces (FvpVideoView) arrive after prepare(), when
        // the video decoder has already opened without a tunnel surface.
        // Re-set the decoder list to force a decoder re-open so the surface
        // property takes effect. Tunneled output requires MediaCodec.
        player->setDecoders(mdk::MediaType::Video, {"AMediaCodec"});
    } else if (const char* ds = getenv("FVP_DIRECT_SURFACE"); ds && ds[0]) {
        // Bench experiments (PR #379 discussion): hand the platform view's
        // surface to MediaCodec itself so decoded frames go straight into the
        // SurfaceView buffer queue — no GL renderer, no EGLConfig, no GPU
        // copy. image=0 disables the AImageReader (frame readback) path. The
        // surface arrives after prepare(), so setDecoders forces a decoder
        // re-open with the surface attached.
        //   FVP_DIRECT_SURFACE=1      -> decoder property surface=<jobject>
        //   FVP_DIRECT_SURFACE=tunnel -> tunnel=1:window=<ANativeWindow*>,
        //                                true tunneled (sideband) playback per
        //                                the AMediaCodec property docs.
        // dv=1 is required for Dolby Vision profile 5 straight to a
        // SurfaceView on SDKs before it became the default.
        player->surface = env->NewGlobalRef(surface);
        player->directSurface = true;
        std::string vd;
        if (std::string(ds) == "tunnel") {
            player->window = ANativeWindow_fromSurface(env, surface);
            vd = "AMediaCodec:image=0:tunnel=1:window=" + std::to_string((intptr_t)player->window);
        } else {
            vd = "AMediaCodec:dv=1:image=0:surface=" + std::to_string((intptr_t)player->surface);
        }
        player->setDecoders(mdk::MediaType::Video, {vd});
    } else {
        if (no10BitConformantWindowConfig()) {
            mdk::GLRenderAPI ra{};
            ra.depth = 8;
            player->setRenderAPI(&ra, surface);
        }
        // Global ref: the surface has to stay valid for later size updates
        // (surfaceChanged) and for the renderer, which outlives this call.
        player->surface = env->NewGlobalRef(surface);
        player->updateNativeSurface(player->surface, w, h);
        player->vo_opaque = player->surface;
    }
    player->width = w;
    player->height = h;
    players[tex_id] = player;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_mediadevkit_fvp_FvpPlugin_nativeSetSurfaceSize(JNIEnv *env, jobject thiz, jlong tex_id,
                                                        jint w, jint h) {
    auto it = players.find(tex_id);
    if (it == players.end()) {
        return;
    }
    auto& player = it->second;
    player->width = w;
    player->height = h;
    // In direct-surface mode the decoder owns the buffer geometry — the
    // compositor scales its layer to the view, so a resize needs nothing here.
    // The GL renderer draws at the surface size and does need it.
    if (!player->directSurface && player->surface) {
        player->updateNativeSurface(player->surface, w, h);
    }
}

extern "C"
JNIEXPORT bool JNICALL
MdkIsEmulator()
{
    // run getprop to see all properties
    char v[PROP_VALUE_MAX+1];
    __system_property_get("ro.kernel.qemu", v);
    if (atoi(v) == 1)
        return true;
    __system_property_get("ro.boot.qemu", v);
    if (atoi(v) == 1)
        return true;
    __system_property_get("ro.product.board", v);
    if (strstr(v, "goldfish"))
        return true;
    __system_property_get("ro.hardware.egl", v);
    if (strstr(v, "emulation"))
        return true;
    __system_property_get("ro.hardware", v);
    if (strstr(v, "ranchu"))
        return true;
    __system_property_get("ro.build.characteristics", v);
    if (strstr(v, "emulator"))
        return true;
    return false;
}

extern "C"
JNIEXPORT void* JNICALL
MdkGetPlayerVid(int64_t tex_id)
{
    if (tex_id < 0)
        return nullptr;
    if (const auto it = players.find(tex_id); it != players.end()) {
        const auto& player = it->second;
        return player->vo_opaque;
    }
    return nullptr;
}