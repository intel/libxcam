/*
 * gl_texture.h - GL texture
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

#ifndef XCAM_GL_TEXTRUE_H
#define XCAM_GL_TEXTRUE_H

#include <gles/gles_std.h>
#include <gles/egl/egl_utils.h>

#include <video_buffer.h>

#define XCAM_GL_MAX_COMPONENTS 4

namespace XCam {

struct GLTextureDesc {
    uint32_t        format;
    uint32_t        width;
    uint32_t        height;
    uint32_t        aligned_width;
    uint32_t        aligned_height;
    uint32_t        size;
    uint32_t        strides[XCAM_GL_MAX_COMPONENTS];
    uint32_t        offsets[XCAM_GL_MAX_COMPONENTS];
    uint32_t        slice_size[XCAM_GL_MAX_COMPONENTS];

    GLTextureDesc ();
};

class GLTexture
{
public:
    ~GLTexture ();

    static SmartPtr<GLTexture> create_texture (
        const GLvoid *data = NULL,
        uint32_t width = 0, uint32_t height = 0, uint32_t format = 0,
        GLenum target = GL_TEXTURE_2D, GLenum usage = GL_STATIC_DRAW);

    static SmartPtr<GLTexture> create_texture (
        SmartPtr<VideoBuffer> &buf,
        GLenum target = GL_TEXTURE_2D, GLenum usage = GL_STATIC_DRAW);

    static XCamReturn destroy_texture (SmartPtr<GLTexture>& tex);

    GLuint get_texture_id () const {
        return _texture_id;
    }
    GLenum get_target () const {
        return _target;
    }
    GLenum get_usage () const {
        return _usage;
    }
    uint32_t get_format () const {
        return _format;
    }
    uint32_t get_width () const {
        return _width;
    }
    uint32_t get_height () const {
        return _height;
    }
    void set_texture_desc (const GLTextureDesc &desc) {
        _desc = desc;
    }
    const GLTextureDesc &get_texture_desc () {
        return _desc;
    }

    XCamReturn bind (uint32_t index);

    XCamReturn bind_image (GLuint unit, GLenum access = GL_READ_WRITE, GLenum format = GL_R8);

    void dump_texture_image (const char *file_name);

private:
    explicit GLTexture (uint32_t width, uint32_t height, uint32_t format, GLuint id, GLenum target = GL_TEXTURE_2D, GLenum usage = GL_STATIC_DRAW);

private:
    XCAM_DEAD_COPY (GLTexture);

private:
    uint32_t       _width;
    uint32_t       _height;
    uint32_t       _format;
    GLuint         _texture_id;
    GLenum         _target;
    GLenum         _usage;
    GLTextureDesc  _desc;
    static EGLImage _egl_image;
};

}

#endif  //XCAM_GL_TEXTURE_H
