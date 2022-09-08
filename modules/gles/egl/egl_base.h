/*
 * egl_base.h - EGL base class
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_EGL_BASE_H
#define XCAM_EGL_BASE_H

#include <gles/egl/egl_utils.h>

#if HAVE_GBM
#include <gbm.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <xcam_mutex.h>

namespace XCam {

class GLTexture;
class VideoBuffer;

class EGLBase {
public:
    ~EGLBase ();
    static SmartPtr<EGLBase> instance ();

    bool init (const char* node_name = NULL);
    bool is_inited () const;

    bool get_display (const char *node_name, EGLDisplay &display);
    bool get_display (NativeDisplayType native_display, EGLDisplay &display);
    bool initialize (EGLDisplay display, EGLint *major, EGLint *minor);
    bool choose_config (
        EGLDisplay display, EGLint const *attribs, EGLConfig *configs,
        EGLint config_size, EGLint *num_config);
    bool create_context (
        EGLDisplay display, EGLConfig config, EGLContext share_context, EGLint const *attribs,
        EGLContext &context);
    bool create_window_surface (
        EGLDisplay display, EGLConfig config, NativeWindowType native_window, EGLint const *attribs,
        EGLSurface &surface);
    bool make_current (EGLDisplay display, EGLSurface draw, EGLSurface read, EGLContext context);
    bool swap_buffers (EGLDisplay display, EGLSurface surface);

    bool destroy_context (EGLDisplay display, EGLContext &context);
    bool destroy_surface (EGLDisplay display, EGLSurface &surface);
    bool terminate (EGLDisplay display);

    EGLImage create_image (int dmabuf_fd,
                           EGLuint64KHR modifiers,
                           uint32_t width,
                           uint32_t height,
                           EGLint stride,
                           EGLint offset,
                           int fourcc);

    bool destroy_image (EGLImage image);

    SmartPtr<VideoBuffer> export_dma_buffer (const SmartPtr<GLTexture>& gl_texture);

private:
    EGLBase ();

private:
    static SmartPtr<EGLBase>  _instance;
    static Mutex      _instance_mutex;
    EGLDisplay        _display;
    EGLContext        _context;
    EGLSurface        _surface;
#if HAVE_GBM
    char              *_node_name;
    gbm_device        *_gbm_device;
    int32_t           _device;
#endif
    bool              _inited;

};

}

#endif // XCAM_EGL_BASE_H
