/*
 * gl_texture.cpp - GL texture
 *
 *  Copyright (c) 2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "gl_texture.h"
#include <gles/egl/egl_utils.h>
#include <gles/egl/egl_base.h>

#include <dma_video_buffer.h>

namespace XCam {

GLTextureDesc::GLTextureDesc ()
    : format (V4L2_PIX_FMT_NV12)
    , width (0)
    , height (0)
    , aligned_width (0)
    , aligned_height (0)
    , size (0)
{
    xcam_mem_clear (strides);
    xcam_mem_clear (slice_size);
    xcam_mem_clear (offsets);
}

GLTexture::GLTexture (uint32_t width, uint32_t height, uint32_t format, GLuint id, GLenum target, GLenum usage)
    : _width (width)
    , _height (height)
    , _format (format)
    , _texture_id (id)
    , _target (target)
    , _usage (usage)
{
}

XCamReturn
GLTexture::bind (uint32_t index)
{
    XCAM_UNUSED (index);

    glBindTexture (_target, _texture_id);
    GLenum error = gl_error ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GL bind texture:%d failed, error flag: %s", _texture_id, gl_error_string (error));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLTexture::bind_image (GLuint index, GLenum access, GLenum format)
{
    glBindImageTexture (index, /* unit, note that we're not offsetting GL_TEXTURE0 */
                        _texture_id, 0, GL_FALSE, 0, access, format);
    GLenum error = gl_error ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GL bind texture:%d failed, error flag: %s", _texture_id, gl_error_string (error));
    return XCAM_RETURN_NO_ERROR;
}

GLTexture::~GLTexture ()
{
    XCAM_LOG_DEBUG ("GLTexture::~GLTexture");
    if (_texture_id) {
        glDeleteTextures (1, &_texture_id);

        GLenum error = gl_error ();
        if (error != GL_NO_ERROR) {
            XCAM_LOG_WARNING (
                "GL Texture delete failed, error flag: %s", gl_error_string (error));
        }
    }
}

SmartPtr<GLTexture>
GLTexture::create_texture (
    const GLvoid *data,
    uint32_t width,
    uint32_t height,
    uint32_t format,
    GLenum target,
    GLenum usage)
{
    XCAM_ASSERT (width > 0);
    XCAM_ASSERT (height > 0);
    XCAM_LOG_DEBUG ("GLTexture::create_texture from buffer: width:%d, height:%d, format:%s", width, height, xcam_fourcc_to_string (format));

    if (format != V4L2_PIX_FMT_NV12 && format != V4L2_PIX_FMT_YUV420) {
        XCAM_LOG_ERROR ("invaild input image format");
        return NULL;
    }

    GLenum error;
    GLuint texture_id = 0;
    glGenTextures (1, &texture_id);
    error = gl_error ();
    XCAM_FAIL_RETURN (
        ERROR, texture_id && (error == GL_NO_ERROR), NULL,
        "GL texture creation failed in glGenTextures, error flag: %s", gl_error_string (error));

    glBindTexture (target, texture_id);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "GL texture creation failed in glBindTexture:%d, error flag: %s",
        texture_id, gl_error_string (error));

    glTexImage2D (target, 0, GL_RED, width, height * 3 / 2, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "GL texture creation failed in glTexImage2D, id:%d, error flag: %s",
        texture_id, gl_error_string (error));

    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "GL texture creation failed in glTexParameteri, id:%d, error flag: %s",
        texture_id, gl_error_string (error));

    return new GLTexture (width, height, format, texture_id, target, usage);
}

EGLImage GLTexture::_egl_image;

SmartPtr<GLTexture>
GLTexture::create_texture (
    SmartPtr<VideoBuffer> &buf,
    GLenum target,
    GLenum usage)
{
    SmartPtr<DmaVideoBuffer> dma_buf = buf.dynamic_cast_ptr<DmaVideoBuffer> ();
    const VideoBufferInfo &info = dma_buf->get_video_info ();

    XCAM_LOG_DEBUG ("GLTexture::create_texture DmaVideoBuffer width:%d, height:%d, stride:%d, offset:%d, format:%s", info.width, info.height, info.strides[0], info.offsets[0], xcam_fourcc_to_string (info.format));
    XCAM_LOG_DEBUG ("modifiers:%lu, dmabuf fd:%d, fourcc:%s", info.modifiers[0], dma_buf->get_fd (), xcam_fourcc_to_string(info.fourcc));

    SmartPtr<EGLBase> egl_base = EGLBase::instance ();

    int dmabuf_fd = dma_buf->get_fd ();
    uint32_t width = info.width;
    uint32_t height = info.height;
    uint32_t format = info.format;
    uint32_t strides = info.strides[0];
    uint32_t offsets = info.offsets[0];
    uint64_t modifiers = info.modifiers[0];
    uint32_t fourcc = info.fourcc;

    GLuint texture_id = 0;
    if (false == egl_base->destroy_image (_egl_image)) {
        //XCAM_LOG_WARNING ("destroy egl image failed!");
    }
    _egl_image = egl_base->create_image (
                     dmabuf_fd,
                     modifiers,
                     width,
                     height * 3 / 2,
                     strides,
                     offsets,
                     fourcc);

    XCAM_ASSERT (_egl_image != EGL_NO_IMAGE);

    glGenTextures (1, &texture_id);
    GLenum error = gl_error ();
    XCAM_FAIL_RETURN (
        ERROR, texture_id && (error == GL_NO_ERROR), NULL,
        "GL texture creation failed, error flag: %s", gl_error_string (error));

    glActiveTexture (GL_TEXTURE0);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "GL texture active failed when bind texture:%d, error flag: %s",
        texture_id, gl_error_string (error));

    glBindTexture (target, texture_id);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "GL texture creation failed when bind texture:%d, error flag: %s",
        texture_id, gl_error_string (error));

    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress ("glEGLImageTargetTexture2DOES");

    glEGLImageTargetTexture2DOES (target, _egl_image);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "EGL image target texture2D:%d, error flag: %s",
        texture_id, gl_error_string (error));

    glTexParameteri (target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "GL texture parameter:%d, error flag: %s",
        texture_id, gl_error_string (error));

    glTexParameteri (target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    XCAM_FAIL_RETURN (
        ERROR, (error = gl_error ()) == GL_NO_ERROR, NULL,
        "GL texture parameter:%d, error flag: %s",
        texture_id, gl_error_string (error));

    return new GLTexture (width, height, format, texture_id, target, usage);
}

XCamReturn
GLTexture::destroy_texture (SmartPtr<GLTexture>& tex)
{
    if (tex.ptr ()) {
        tex.release ();
    }

    SmartPtr<EGLBase> egl_base = EGLBase::instance ();

    if (false == egl_base->destroy_image (_egl_image)) {
        XCAM_LOG_WARNING ("destroy egl image failed!");
        return XCAM_RETURN_ERROR_EGL;
    }

    return XCAM_RETURN_NO_ERROR;
}

void
GLTexture::dump_texture_image (const char *file_name)
{
    uint32_t width = get_width ();
    uint32_t height = get_height ();
    uint32_t format = get_format ();

    uint8_t* texture_data = NULL;
    if (V4L2_PIX_FMT_NV12 == format) {
        texture_data = new uint8_t [width * height * 3 / 2];
    }
    XCAM_LOG_DEBUG ("image width:%d, height:%d, format:%s", width, height, xcam_fourcc_to_string (format));

    glActiveTexture (GL_TEXTURE0);
    GLenum error = gl_error ();
    if (error != GL_NO_ERROR) {
        XCAM_LOG_ERROR ("Error glActiveTexture, id:%d, error flag: %s", _texture_id, gl_error_string (error));
    }

    glBindTexture (_target, _texture_id);
    error = gl_error ();
    if (error != GL_NO_ERROR) {
        XCAM_LOG_ERROR ("Error glBindTexture, id:%d, error flag: %s", _texture_id, gl_error_string (error));
    }

    GLuint fbo_id;
    glGenFramebuffers (1, &fbo_id);
    error = gl_error ();
    if (error != GL_NO_ERROR) {
        XCAM_LOG_ERROR ("Error glGenFramebuffers, id:%d, error flag: %s", _texture_id, gl_error_string (error));
    }

    glBindFramebuffer (GL_FRAMEBUFFER, fbo_id);
    error = gl_error ();
    if (error != GL_NO_ERROR) {
        XCAM_LOG_ERROR ("Error glBindFramebuffer, id:%d, error flag: %s", _texture_id, gl_error_string (error));
    }

    glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture_id, 0);
    error = gl_error ();
    if (error != GL_NO_ERROR) {
        XCAM_LOG_ERROR ("Error glFramebufferTexture2D, id:%d, error flag: %s", _texture_id, gl_error_string (error));
    }

    glReadPixels (0, 0, width, height * 3 / 2, GL_RED, GL_UNSIGNED_BYTE, texture_data);
    error = gl_error ();
    if (error != GL_NO_ERROR) {
        XCAM_LOG_ERROR ("Error glReadPixels, id:%d, error flag: %s", _texture_id, gl_error_string (error));
    }

    FILE* fbo_file = fopen (file_name, "wb");
    fwrite (texture_data, height * width * 3 / 2, 1, fbo_file);
    fclose (fbo_file);
    delete [] texture_data;
}

}

