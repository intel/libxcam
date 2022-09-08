/*
 * gl_dma_buffer_handler.cpp - gl DMA buffer handler implementation
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

#include "gl_dma_buffer_handler.h"
#include "gl_utils.h"
#include "gl_sync.h"

#define INVALID_INDEX (uint32_t)(-1)

namespace XCam {

static const GLShaderInfo shaders_info[] = {
    {
        GL_COMPUTE_SHADER,
        "shader_copy_tex_to_ssbo",
#include "shader_copy_tex_to_ssbo.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_copy_ssbo_to_tex",
#include "shader_copy_ssbo_to_tex.comp.slx"
        , 0
    }
};

namespace GLDmaBufferPriv {

static SmartPtr<GLImageShader>
create_shader (ShaderID id, const char *prog_name)
{
    SmartPtr<GLImageShader> shader = new GLImageShader (shaders_info[id].name);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shaders_info[id], prog_name);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, NULL,
        "gl-dmabuf create compute program for %s failed", shaders_info[id].name);

    return shader;
}

class Impl
{
public:
    explicit Impl (GLDmaBufferHandler *handler);
    virtual ~Impl ();

    virtual XCamReturn configure_resource (const SmartPtr<ImageHandler::Parameters> &param) = 0;
    virtual XCamReturn start (const SmartPtr<ImageHandler::Parameters> &param) = 0;

public:
    void dump_texture_image (const char* prefix, int32_t idx);

protected:
    GLDmaBufferHandler        *_handler;
    SmartPtr<GLImageShader>    _shader;
    SmartPtr<GLTexture>        _texture;
};

class DmaBufferReader
    : public Impl
{
public:
    explicit DmaBufferReader (GLDmaBufferHandler *handler);
    virtual ~DmaBufferReader () {}

    virtual XCamReturn configure_resource (const SmartPtr<ImageHandler::Parameters> &param);
    virtual XCamReturn start (const SmartPtr<ImageHandler::Parameters> &param);
};

class DmaBufferWriter
    : public Impl
{
public:
    explicit DmaBufferWriter (GLDmaBufferHandler *handler);
    virtual ~DmaBufferWriter () {}

    virtual XCamReturn configure_resource (const SmartPtr<ImageHandler::Parameters> &param);
    virtual XCamReturn start (const SmartPtr<ImageHandler::Parameters> &param);
};

Impl::Impl (GLDmaBufferHandler *handler)
    : _handler (handler)
{
}

Impl::~Impl ()
{
    _shader.release ();
    GLTexture::destroy_texture (_texture);
    _texture.release ();
}

void
Impl::dump_texture_image (const char* prefix, int32_t idx)
{
    char file_name[256];

    snprintf ( file_name, 256, "%s-texture-%d.yuv", prefix,  idx);
    _texture->dump_texture_image (file_name);
}

DmaBufferReader::DmaBufferReader (GLDmaBufferHandler *handler)
    : Impl (handler)
{
}

XCamReturn
DmaBufferReader::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    _shader = create_shader (CopyTex2SSBO, "shader_copy_tex_to_ssbo");
    XCAM_ASSERT (_shader.ptr ());

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &out_info = param->out_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, out_info.width > 0, XCAM_RETURN_ERROR_PARAM,
        "gl-dmabuf invalid output width: %d", out_info.width);

    const size_t uint_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.aligned_width / uint_bytes;
    uint32_t in_img_height = in_info.aligned_height * 3 / 2;
    uint32_t out_img_width = out_info.aligned_width / uint_bytes;
    uint32_t out_img_height = out_info.aligned_height * 3 / 2;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    _shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (out_img_width, 8);
    groups_size.y = XCAM_ALIGN_UP (out_img_height, 8);
    groups_size.z = 1;

    _shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DmaBufferReader::start (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<VideoBuffer> dma_buf = param->in_buf;
    SmartPtr<GLBuffer> gl_buf = get_glbuffer (param->out_buf);

    _texture = GLTexture::create_texture (dma_buf);
    if (!_texture.ptr ()) {
        return XCAM_RETURN_ERROR_GLES;
    }
#if DUMP_TEST
    _texture->dump_texture_image ("dmabuf_to_shader_input_texture.yuv");
#endif

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindTexture (_texture, 0));
    cmds.push_back (new GLCmdBindBufRange (gl_buf, 1));
    _shader->set_commands (cmds);

    return _shader->work (NULL);
}

DmaBufferWriter::DmaBufferWriter (GLDmaBufferHandler *handler)
    : Impl (handler)
{
}

XCamReturn
DmaBufferWriter::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    _shader = create_shader (CopySSBO2Tex, "shader_copy_ssbo_to_tex");
    XCAM_ASSERT (_shader.ptr ());

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &out_info = param->out_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, out_info.width > 0, XCAM_RETURN_ERROR_PARAM,
        "gl-dmabuf invalid output width: %d", out_info.width);

    const size_t uint_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.aligned_width / uint_bytes;
    uint32_t in_img_height = in_info.aligned_height * 3 / 2;
    uint32_t out_img_width = out_info.aligned_width / uint_bytes;
    uint32_t out_img_height = out_info.aligned_height * 3 / 2;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    _shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (in_img_width, 8);
    groups_size.y = XCAM_ALIGN_UP (in_img_height, 8);
    groups_size.z = 1;
    _shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DmaBufferWriter::start (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<GLBuffer> gl_buf = get_glbuffer (param->in_buf);

    SmartPtr<VideoBuffer> dma_buf = param->out_buf;
    _texture = GLTexture::create_texture (dma_buf);
    if (!_texture.ptr ()) {
        return XCAM_RETURN_ERROR_GLES;
    }

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (gl_buf, 0));
    cmds.push_back (new GLCmdBindTexture (_texture, 1));

    _shader->set_commands (cmds);

    return _shader->work (NULL);
}

} //GLDmaBufferPriv

GLDmaBufferHandler::GLDmaBufferHandler (const char *name)
    : GLImageHandler (name)
{
    _opt_type = CopyTex2SSBO;
}

GLDmaBufferHandler::~GLDmaBufferHandler ()
{
    terminate ();
}

XCamReturn
GLDmaBufferHandler::set_opt_type (const uint32_t type)
{
    _opt_type = type;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLDmaBufferHandler::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SmartPtr<GLDmaBufferPriv::Impl> impl;
    switch (_opt_type) {
    case CopyTex2SSBO:
        impl = new GLDmaBufferPriv::DmaBufferReader (this);
        break;
    case CopySSBO2Tex:
        impl = new GLDmaBufferPriv::DmaBufferWriter (this);
        break;
    }
    XCAM_ASSERT (impl.ptr ());

    ret = impl->configure_resource (param);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-dmabuf configure resource failed");

    _impl = impl;
    return ret;
}

XCamReturn
GLDmaBufferHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    return _impl->start (param);
};

XCamReturn
GLDmaBufferHandler::terminate ()
{
    if (_impl.ptr ()) {
        _impl.release ();
    }

    return GLImageHandler::terminate ();
}

XCamReturn
GLDmaBufferHandler::read_dma_buffer (const SmartPtr<DmaVideoBuffer> &dma_buf, SmartPtr<VideoBuffer> &out_buf)
{
    XCAM_ASSERT (dma_buf.ptr ());
    XCAM_ASSERT (out_buf.ptr ());

    XCamReturn ret;
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (dma_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    SmartPtr<GLDmaBufferPriv::Impl> impl = new GLDmaBufferPriv::DmaBufferReader (this);
    XCAM_ASSERT (impl.ptr ());

    ret = impl->configure_resource (param);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-dmabuf reader configure resource failed");

    ret = impl->start (param);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-dmabuf reader execute failed");

    GLSync::flush ();
    GLSync::finish ();
    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

XCamReturn
GLDmaBufferHandler::write_dma_buffer (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<DmaVideoBuffer> &dma_buf)
{
    XCAM_ASSERT (in_buf.ptr ());
    XCAM_ASSERT (dma_buf.ptr ());
    XCAM_ASSERT (-1 != dma_buf->get_fd ());

    XCamReturn ret;
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, dma_buf);
    XCAM_ASSERT (param.ptr ());

    SmartPtr<GLDmaBufferPriv::Impl> impl = new GLDmaBufferPriv::DmaBufferWriter (this);
    XCAM_ASSERT (impl.ptr ());

    ret = impl->configure_resource (param);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-dmabuf writer configure resource failed");

    ret = impl->start (param);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-dmabuf writer execute failed");

    GLSync::flush ();
    GLSync::finish ();

    return ret;
}

}
