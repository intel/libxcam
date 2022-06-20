/*
 * egl_base.cpp - EGL base implementation
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
 * Author: Wei Zong <wei.zong@intel.com>
 */

#include "egl_base.h"

#include "gl_texture.h"

#include <dma_video_buffer.h>

namespace XCam {

SmartPtr<EGLBase> EGLBase::_instance;
Mutex EGLBase::_instance_mutex;

SmartPtr<EGLBase>
EGLBase::instance ()
{
    SmartLock locker(_instance_mutex);
    if (_instance.ptr())
        return _instance;

    _instance = new EGLBase ();

    return _instance;
}

EGLBase::EGLBase ()
    : _display (EGL_NO_DISPLAY)
    , _context (EGL_NO_CONTEXT)
    , _surface (EGL_NO_SURFACE)
#if HAVE_GBM
    , _node_name (NULL)
    , _gbm_device (NULL)
    , _device (0)
#endif
    , _inited (false)
{
}

EGLBase::~EGLBase ()
{
    if (_display != EGL_NO_DISPLAY) {
        XCAM_LOG_DEBUG ("EGLBase::~EGLBase distroy display:%d\n", _display);

        if (_context != EGL_NO_CONTEXT) {
            destroy_context (_display, _context);
            _context = EGL_NO_CONTEXT;
        }

        if (_surface != EGL_NO_SURFACE) {
            destroy_surface (_display, _surface);
            _surface = EGL_NO_SURFACE;
        }

        terminate (_display);
        _display = EGL_NO_DISPLAY;

#if HAVE_GBM
        if (NULL != _gbm_device) {
            gbm_device_destroy (_gbm_device);
        }
        if (NULL != _node_name) {
            xcam_free (_node_name);
        }
        close (_device);
#endif
    }
}

bool
EGLBase::init (const char* node_name)
{
    if (_inited) {
        XCAM_LOG_WARNING ("EGLBase::init already inited! \n");
        return true;
    }

    bool ret = false;
    if (NULL != node_name) {
        XCAM_LOG_DEBUG ("EGL init: %s", node_name);
        ret = get_display (node_name, _display);
        XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: get display failed");
    } else {
        ret = get_display (EGL_DEFAULT_DISPLAY, _display);
        XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: get display failed");
    }

    EGLint major, minor;
    ret = initialize (_display, &major, &minor);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: EGL initialize failed");
    XCAM_LOG_DEBUG ("EGL version: %d.%d", major, minor);

    EGLConfig configs;
    EGLint num_config;
    EGLint cfg_attribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE};
    ret = choose_config (_display, cfg_attribs, &configs, 1, &num_config);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: choose config failed");

    EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    ret = create_context (_display, configs, EGL_NO_CONTEXT, ctx_attribs, _context);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: create context failed");

    ret = make_current (_display, _surface, _surface, _context);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: make current failed");

    _inited = true;

    return true;
}

bool
EGLBase::is_inited () const
{
    return _inited;
}

bool
EGLBase::get_display (const char *node_name, EGLDisplay &display)
{
#if HAVE_GBM
    if (!node_name) {
        XCAM_LOG_ERROR ("get disply device node name is NULL!");
        return false;
    }
    _node_name = strndup (node_name, XCAM_MAX_STR_SIZE);
    XCAM_FAIL_RETURN (ERROR, _node_name != NULL, false, "EGLInit: copy gbm device name failed");

    _device = open (_node_name, O_RDWR);
    XCAM_FAIL_RETURN (ERROR, _device > 0, false, "EGLInit: EGL open device node:%s failed", _node_name);

    _gbm_device = gbm_create_device (_device);
    XCAM_FAIL_RETURN (ERROR, _gbm_device != NULL, false, "EGLInit: EGL create gbm device failed");

    display = eglGetPlatformDisplay (EGL_PLATFORM_GBM_MESA, _gbm_device, NULL);
    XCAM_FAIL_RETURN (
        ERROR, display != EGL_NO_DISPLAY, false,
        "EGLInit: get display failed");
    return true;
#else
    XCAM_UNUSED (node_name);
    display = eglGetDisplay (EGL_DEFAULT_DISPLAY);
    XCAM_FAIL_RETURN (
        ERROR, display != EGL_NO_DISPLAY, false,
        "EGLInit: get display failed");
    return true;
#endif
}

bool
EGLBase::get_display (NativeDisplayType native_display, EGLDisplay &display)
{
    display = eglGetDisplay (native_display);
    XCAM_FAIL_RETURN (
        ERROR, display != EGL_NO_DISPLAY, false,
        "EGLInit: get display failed");
    return true;
}

bool
EGLBase::initialize (EGLDisplay display, EGLint *major, EGLint *minor)
{
    EGLBoolean ret = eglInitialize (display, major, minor);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: initialize failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::choose_config (
    EGLDisplay display, EGLint const *attribs, EGLConfig *configs,
    EGLint config_size, EGLint *num_config)
{
    EGLBoolean ret = eglChooseConfig (display, attribs, configs, config_size, num_config);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: choose config failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::create_context (
    EGLDisplay display, EGLConfig config, EGLContext share_context, EGLint const *attribs,
    EGLContext &context)
{
    context = eglCreateContext (display, config, share_context, attribs);
    XCAM_FAIL_RETURN (
        ERROR, context != EGL_NO_CONTEXT, false,
        "EGLInit: create context failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::create_window_surface (
    EGLDisplay display, EGLConfig config, NativeWindowType native_window, EGLint const *attribs,
    EGLSurface &surface)
{
    surface = eglCreateWindowSurface (display, config, native_window, attribs);
    XCAM_FAIL_RETURN (
        ERROR, surface != EGL_NO_SURFACE, false,
        "EGLInit: create window surface failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::make_current (EGLDisplay display, EGLSurface draw, EGLSurface read, EGLContext context)
{
    EGLBoolean ret = eglMakeCurrent (display, draw, read, context);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: make current failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::swap_buffers (EGLDisplay display, EGLSurface surface)
{
    EGLBoolean ret = eglSwapBuffers (display, surface);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: swap buffers failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::destroy_context (EGLDisplay display, EGLContext &context)
{
    EGLBoolean ret = eglDestroyContext (display, context);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: destroy context failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    XCAM_FAIL_RETURN (ERROR, ret == EGL_TRUE, false, "EGLInit: destroy context failed");

    return true;
}

bool
EGLBase::destroy_surface (EGLDisplay display, EGLSurface &surface)
{
    EGLBoolean ret = eglDestroySurface (display, surface);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: destroy surface failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    XCAM_FAIL_RETURN (ERROR, ret == EGL_TRUE, false, "EGLInit: destroy surface failed");

    return true;
}

bool
EGLBase::terminate (EGLDisplay display)
{
    EGLBoolean ret = eglTerminate (display);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: terminate failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    XCAM_FAIL_RETURN (ERROR, ret == EGL_TRUE, false, "EGLInit: terminate failed");
    return true;
}

EGLImage
EGLBase::create_image (
    int dmabuf_fd,
    EGLuint64KHR modifiers,
    uint32_t width,
    uint32_t height,
    EGLint stride,
    EGLint offset,
    int fourcc)
{
    EGLAttrib const attribute_list[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_LINUX_DRM_FOURCC_EXT, fourcc,
        EGL_DMA_BUF_PLANE0_FD_EXT, dmabuf_fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, stride,
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (uint32_t)(modifiers & ((((uint64_t)1) << 33) - 1)),
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (uint32_t)((modifiers >> 32) & ((((uint64_t)1) << 33) - 1)),
        EGL_NONE
    };

    EGLImage egl_image = eglCreateImage (_display,
                                         NULL,
                                         EGL_LINUX_DMA_BUF_EXT,
                                         (EGLClientBuffer)NULL,
                                         attribute_list);

    XCAM_ASSERT (egl_image != EGL_NO_IMAGE);

    return egl_image;
}

bool
EGLBase::destroy_image (EGLImage image)
{
    EGLBoolean ret = eglDestroyImage (_display, image);

    return ret;
}

SmartPtr<VideoBuffer>
EGLBase::export_dma_buffer (const SmartPtr<GLTexture>& gl_texture)
{
    int dmabuf_fd;
    int fourcc;
    int num_planes;
    EGLuint64KHR modifiers;
    EGLint stride;
    EGLint offset;

    GLuint tex_id = gl_texture->get_texture_id ();
    uint32_t width = gl_texture->get_width ();
    uint32_t height = gl_texture->get_height ();
    uint32_t format = gl_texture->get_format ();

    // EGL: Create EGL image from the GL texture
    EGLImage egl_image = eglCreateImage (_display,
                                         _context,
                                         EGL_GL_TEXTURE_2D,
                                         (EGLClientBuffer)(uint64_t)tex_id,
                                         NULL);

    XCAM_ASSERT (egl_image != EGL_NO_IMAGE);

    PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC eglExportDMABUFImageQueryMESA =
        (PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC)eglGetProcAddress ("eglExportDMABUFImageQueryMESA");

    EGLBoolean queried = eglExportDMABUFImageQueryMESA (_display,
                         egl_image,
                         &fourcc,
                         &num_planes,
                         &modifiers);

    if (!queried) {
        XCAM_LOG_ERROR ("egl query export DMA buffer image MESA failed!");
        return NULL;
    }

    PFNEGLEXPORTDMABUFIMAGEMESAPROC eglExportDMABUFImageMESA =
        (PFNEGLEXPORTDMABUFIMAGEMESAPROC)eglGetProcAddress ("eglExportDMABUFImageMESA");

    EGLBoolean exported = eglExportDMABUFImageMESA (_display,
                          egl_image,
                          &dmabuf_fd,
                          &stride,
                          &offset);

    if (!exported) {
        XCAM_LOG_ERROR ("egl export DMA buffer image MESA failed!");
        return NULL;
    }

    VideoBufferInfo info;
    info.init (format, width, height);
    info.strides[0] = stride;
    info.offsets[0] = offset;
    info.modifiers[0] = modifiers;
    info.fourcc = fourcc;

    XCAM_LOG_DEBUG ("DMA buffer width:%d, height:%d, stride:%d, offset:%d", info.width, info.height, info.strides[0], info.offsets[0]);
    XCAM_LOG_DEBUG ("  modifiers:%lu, fd:%d, fourcc:%s", info.modifiers[0], dmabuf_fd, xcam_fourcc_to_string(info.fourcc));
    XCAM_LOG_DEBUG ("  foucc:%s", xcam_fourcc_to_string (fourcc));


    SmartPtr<DmaVideoBuffer> dma_buf = new DmaVideoBuffer (info, dmabuf_fd);
    XCAM_ASSERT (dma_buf.ptr ());

    return dma_buf;
}

}
