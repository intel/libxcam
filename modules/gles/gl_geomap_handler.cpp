/*
 * gl_geomap_handler.cpp - gl geometry map handler implementation
 *
 *  Copyright (c) 2018 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "gl_geomap_handler.h"
#include "gl_utils.h"
#include "gl_sync.h"

#define XCAM_GL_GEOMAP_ALIGN_X 4
#define XCAM_GL_GEOMAP_ALIGN_Y 2

namespace XCam {

static const GLShaderInfo shaders_info[] = {
    {
        GL_COMPUTE_SHADER,
        "shader_geomap",
#include "shader_geomap.comp.slx"
      , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_geomap_yuv420",
#include "shader_geomap_yuv420.comp.slx"
      , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_geomap_fastmap_y",
#include "shader_geomap_fastmap_y.comp.slx"
      , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_geomap_fastmap_uv_nv12",
#include "shader_geomap_fastmap_uv_nv12.comp.slx"
      , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_geomap_fastmap_uv_yuv420",
#include "shader_geomap_fastmap_uv_yuv420.comp.slx"
      , 0
    }
};

namespace GLGeoMapPriv {

enum ShaderID {
    ShaderComMapNV12 = 0,    // NV12 common mapping
    ShaderComMapYUV420,      // YUV420 common mapping
    ShaderFastMapY,          // Y planar fast mapping
    ShaderFastMapUVNV12,     // NV12 UV planar fast mapping
    ShaderFastMapUVYUV420    // YUV420 UV planar fast mapping
};

class ComMap
{
public:
    explicit ComMap (GLGeoMapHandler *mapper);
    virtual ~ComMap ();

    virtual XCamReturn start (const SmartPtr<ImageHandler::Parameters> &param) = 0;
    virtual XCamReturn configure_resource (const SmartPtr<ImageHandler::Parameters> &param) = 0;
    virtual XCamReturn prepare_dump_coords () = 0;

protected:
    GLGeoMapHandler           *_mapper;
    SmartPtr<GLImageShader>    _shader;
};

class ComMapNV12
    : public ComMap
{
public:
    explicit ComMapNV12 (GLGeoMapHandler *mapper);
    virtual ~ComMapNV12 () {}

    virtual XCamReturn start (const SmartPtr<ImageHandler::Parameters> &param);
    virtual XCamReturn configure_resource (const SmartPtr<ImageHandler::Parameters> &param);
    virtual XCamReturn prepare_dump_coords ();
};

class ComMapYUV420
    : public ComMap
{
public:
    explicit ComMapYUV420 (GLGeoMapHandler *mapper);
    virtual ~ComMapYUV420 () {}

    virtual XCamReturn start (const SmartPtr<ImageHandler::Parameters> &param);
    virtual XCamReturn configure_resource (const SmartPtr<ImageHandler::Parameters> &param);
    virtual XCamReturn prepare_dump_coords ();
};

class FastMap
{
public:
    explicit FastMap (GLGeoMapHandler *mapper);
    virtual ~FastMap ();

    XCamReturn start (const SmartPtr<ImageHandler::Parameters> &param);
    XCamReturn configure_resource (const SmartPtr<ImageHandler::Parameters> &param);

protected:
    virtual XCamReturn init_shaders () = 0;
    virtual XCamReturn start_y (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf) = 0;
    virtual XCamReturn start_uv (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf) = 0;
    virtual XCamReturn fix_uv_parameters (const VideoBufferInfo &in_info) = 0;

    XCamReturn fix_y_parameters (const VideoBufferInfo &in_info);

protected:
    GLGeoMapHandler           *_mapper;

    SmartPtr<GLImageShader>    _shader_y;
    SmartPtr<GLImageShader>    _shader_uv;
};

class FastMapNV12
    : public FastMap
{
public:
    explicit FastMapNV12 (GLGeoMapHandler *mapper);
    virtual ~FastMapNV12 () {}

private:
    virtual XCamReturn init_shaders ();
    virtual XCamReturn start_y (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn start_uv (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn fix_uv_parameters (const VideoBufferInfo &in_info);
};

class FastMapYUV420
    : public FastMap
{
public:
    explicit FastMapYUV420 (GLGeoMapHandler *mapper);
    virtual ~FastMapYUV420 () {}

private:
    virtual XCamReturn init_shaders ();
    virtual XCamReturn start_y (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn start_uv (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn fix_uv_parameters (const VideoBufferInfo &in_info);
};

ComMap::ComMap (GLGeoMapHandler *mapper)
    : _mapper (mapper)
{
}

ComMap::~ComMap ()
{
    _shader.release ();
}

ComMapNV12::ComMapNV12 (GLGeoMapHandler *mapper)
    : ComMap (mapper)
{
}

XCamReturn
ComMapNV12::start (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<GLBuffer> in_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);
    const SmartPtr<GLBuffer> &lut_buf = _mapper->get_lut_buf ();

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (in_buf, 1, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 2, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 3, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufBase (lut_buf, 4));
    cmds.push_back (new GLCmdUniformTVect<float, 4> ("lut_step", _mapper->get_lut_step ()));
    _shader->set_commands (cmds);

    return _shader->work (NULL);
}

static SmartPtr<GLImageShader>
create_shader (ShaderID id, const char *prog_name)
{
    SmartPtr<GLImageShader> shader = new GLImageShader (shaders_info[id].name);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shaders_info[id], prog_name);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, NULL,
        "gl-geomap create compute program for %s failed", shaders_info[id].name);

    return shader;
}

XCamReturn
ComMapNV12::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    _shader = create_shader (ShaderComMapNV12, "commap_program_nv12");
    XCAM_ASSERT (_shader.ptr ());

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    const GLBufferDesc &lut_desc = _mapper->get_lut_buf ()->get_buffer_desc ();
    const Rect &std_area = _mapper->get_std_area ();

    float factor_x, factor_y;
    _mapper->get_factors (factor_x, factor_y);
    float lut_std_step[2] = {1.0f / factor_x, 1.0f / factor_y};

    uint32_t width, height, std_width, std_height;
    _mapper->get_output_size (width, height);
    _mapper->get_std_output_size (std_width, std_height);

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.width / unit_bytes;
    uint32_t out_img_width = width / unit_bytes;
    uint32_t extended_offset = _mapper->get_extended_offset () / unit_bytes;

    std_width /= unit_bytes;
    uint32_t std_offset = std_area.pos_x / unit_bytes;
    uint32_t std_valid_width = std_area.width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", in_info.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("extended_offset", extended_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_width", std_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_height", std_area.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_offset", std_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_width", lut_desc.width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_height", lut_desc.height));
    cmds.push_back (new GLCmdUniformTVect<float, 2> ("lut_std_step", lut_std_step));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("dump_coords", 0));
    _shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (std_valid_width, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (std_area.height, 8) / 8;
    groups_size.z = 1;
    _shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ComMapNV12::prepare_dump_coords ()
{
    const Rect &std_area = _mapper->get_std_area ();

    GLBufferDesc desc;
    desc.width = std_area.width;
    desc.height = std_area.height;
    desc.size = desc.width * desc.height * sizeof (float);

    SmartPtr<GLBuffer> coordx_y = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    SmartPtr<GLBuffer> coordy_y = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    XCAM_ASSERT (coordx_y.ptr () && coordy_y.ptr ());

    coordx_y->set_buffer_desc (desc);
    coordy_y->set_buffer_desc (desc);
    _mapper->set_coordx_y (coordx_y);
    _mapper->set_coordy_y (coordy_y);

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("dump_coords", 1));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width", desc.width / sizeof (uint32_t)));
    cmds.push_back (new GLCmdBindBufBase (coordx_y, 5));
    cmds.push_back (new GLCmdBindBufBase (coordy_y, 6));

    desc.width /= 2;
    desc.height /= 2;
    desc.size = desc.width * desc.height * sizeof (float);

    SmartPtr<GLBuffer> coordx_uv = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    SmartPtr<GLBuffer> coordy_uv = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    XCAM_ASSERT (coordx_uv.ptr () && coordy_uv.ptr ());

    coordx_uv->set_buffer_desc (desc);
    coordy_uv->set_buffer_desc (desc);
    _mapper->set_coordx_uv (coordx_uv);
    _mapper->set_coordy_uv (coordy_uv);

    cmds.push_back (new GLCmdBindBufBase (coordx_uv, 7));
    cmds.push_back (new GLCmdBindBufBase (coordy_uv, 8));

    _shader->set_commands (cmds);

    return XCAM_RETURN_NO_ERROR;
}

ComMapYUV420::ComMapYUV420 (GLGeoMapHandler *mapper)
    : ComMap (mapper)
{
}

XCamReturn
ComMapYUV420::start (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<GLBuffer> in_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);
    const SmartPtr<GLBuffer> &lut_buf = _mapper->get_lut_buf ();

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0, YUV420PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (in_buf, 1, YUV420PlaneUIdx));
    cmds.push_back (new GLCmdBindBufRange (in_buf, 2, YUV420PlaneVIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 3, YUV420PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 4, YUV420PlaneUIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 5, YUV420PlaneVIdx));
    cmds.push_back (new GLCmdBindBufBase (lut_buf, 6));
    cmds.push_back (new GLCmdUniformTVect<float, 4> ("lut_step", _mapper->get_lut_step ()));
    _shader->set_commands (cmds);

    return _shader->work (NULL);
}

XCamReturn
ComMapYUV420::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    _shader = create_shader (ShaderComMapYUV420, "commap_program_yuv420");
    XCAM_ASSERT (_shader.ptr ());

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    const GLBufferDesc &lut_desc = _mapper->get_lut_buf ()->get_buffer_desc ();
    const Rect &std_area = _mapper->get_std_area ();

    float factor_x, factor_y;
    _mapper->get_factors (factor_x, factor_y);
    float lut_std_step[2] = {1.0f / factor_x, 1.0f / factor_y};

    uint32_t width, height, std_width, std_height;
    _mapper->get_output_size (width, height);
    _mapper->get_std_output_size (std_width, std_height);

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.width / unit_bytes;
    uint32_t out_img_width = width / unit_bytes;
    uint32_t extended_offset = _mapper->get_extended_offset () / unit_bytes;

    std_width /= unit_bytes;
    uint32_t std_offset = std_area.pos_x / unit_bytes;
    uint32_t std_valid_width = std_area.width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", in_info.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("extended_offset", extended_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_width", std_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_height", std_area.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_offset", std_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_width", lut_desc.width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_height", lut_desc.height));
    cmds.push_back (new GLCmdUniformTVect<float, 2> ("lut_std_step", lut_std_step));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("dump_coords", 0));
    _shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (std_valid_width, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (std_area.height, 8) / 8;
    groups_size.z = 1;
    _shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ComMapYUV420::prepare_dump_coords ()
{
    const Rect &std_area = _mapper->get_std_area ();

    GLBufferDesc desc;
    desc.width = std_area.width;
    desc.height = std_area.height;
    desc.size = desc.width * desc.height * sizeof (float);

    SmartPtr<GLBuffer> coordx_y = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    SmartPtr<GLBuffer> coordy_y = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    XCAM_ASSERT (coordx_y.ptr () && coordy_y.ptr ());

    coordx_y->set_buffer_desc (desc);
    coordy_y->set_buffer_desc (desc);
    _mapper->set_coordx_y (coordx_y);
    _mapper->set_coordy_y (coordy_y);

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("dump_coords", 1));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width", desc.width / sizeof (uint32_t)));
    cmds.push_back (new GLCmdBindBufBase (coordx_y, 7));
    cmds.push_back (new GLCmdBindBufBase (coordy_y, 8));

    desc.width /= 2;
    desc.height /= 2;
    desc.size = desc.width * desc.height * sizeof (float);

    SmartPtr<GLBuffer> coordx_uv = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    SmartPtr<GLBuffer> coordy_uv = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    XCAM_ASSERT (coordx_uv.ptr () && coordy_uv.ptr ());

    coordx_uv->set_buffer_desc (desc);
    coordy_uv->set_buffer_desc (desc);
    _mapper->set_coordx_uv (coordx_uv);
    _mapper->set_coordy_uv (coordy_uv);

    cmds.push_back (new GLCmdBindBufBase (coordx_uv, 9));
    cmds.push_back (new GLCmdBindBufBase (coordy_uv, 10));

    _shader->set_commands (cmds);

    return XCAM_RETURN_NO_ERROR;
}

FastMap::FastMap (GLGeoMapHandler *mapper)
    : _mapper (mapper)
{
}

FastMap::~FastMap ()
{
    _shader_y.release ();
    _shader_uv.release ();
}

XCamReturn
FastMap::start (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<GLBuffer> in_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);

    XCamReturn ret = start_y (in_buf, out_buf);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret, "gl-geomap start Y failed");

    ret = start_uv (in_buf, out_buf);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret, "gl-geomap start UV failed");

    return XCAM_RETURN_NO_ERROR;
};

XCamReturn
FastMap::configure_resource (const SmartPtr<ImageHandler::Parameters> &param)
{
    init_shaders ();

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    fix_y_parameters (in_info);
    fix_uv_parameters (in_info);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
FastMap::fix_y_parameters (const VideoBufferInfo &in_info)
{
    const SmartPtr<GLBuffer> &coordx_y = _mapper->get_coordx_y ();
    const GLBufferDesc &desc = coordx_y->get_buffer_desc ();
    const Rect &std_area = _mapper->get_std_area ();

    uint32_t width, height;
    _mapper->get_output_size (width, height);

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.width / unit_bytes;
    uint32_t out_img_width = width / unit_bytes;
    uint32_t extended_offset = _mapper->get_extended_offset () / unit_bytes;
    uint32_t std_valid_width = std_area.width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("extended_offset", extended_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width", desc.width / unit_bytes));
    _shader_y->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (std_valid_width, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (std_area.height, 8) / 8;
    groups_size.z = 1;
    _shader_y->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

FastMapNV12::FastMapNV12 (GLGeoMapHandler *mapper)
    : FastMap (mapper)
{
}

XCamReturn
FastMapNV12::init_shaders ()
{
    _shader_y = create_shader (ShaderFastMapY, "fastmap_program_nv12_y");
    _shader_uv = create_shader (ShaderFastMapUVNV12, "fastmap_program_nv12_uv");
    XCAM_ASSERT (_shader_y.ptr () && _shader_uv.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
FastMapNV12::start_y (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf)
{
    const SmartPtr<GLBuffer> &coordx_y = _mapper->get_coordx_y ();
    const SmartPtr<GLBuffer> &coordy_y = _mapper->get_coordy_y ();

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 1, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (coordx_y, 2));
    cmds.push_back (new GLCmdBindBufRange (coordy_y, 3));
    _shader_y->set_commands (cmds);

    return _shader_y->work (NULL);
}

XCamReturn
FastMapNV12::start_uv (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf)
{
    const SmartPtr<GLBuffer> &coordx_uv = _mapper->get_coordx_uv ();
    const SmartPtr<GLBuffer> &coordy_uv = _mapper->get_coordy_uv ();

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 1, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (coordx_uv, 2));
    cmds.push_back (new GLCmdBindBufRange (coordy_uv, 3));
    _shader_uv->set_commands (cmds);

    return _shader_uv->work (NULL);
}

XCamReturn
FastMapNV12::fix_uv_parameters (const VideoBufferInfo &in_info)
{
    const SmartPtr<GLBuffer> &coordx_uv = _mapper->get_coordx_uv ();
    const GLBufferDesc &desc = coordx_uv->get_buffer_desc ();
    const Rect &std_area = _mapper->get_std_area ();

    uint32_t width, height;
    _mapper->get_output_size (width, height);

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.width / unit_bytes;
    uint32_t out_img_width = (width / 2) / unit_bytes;
    uint32_t extended_offset = (_mapper->get_extended_offset () / 2) / unit_bytes;
    uint32_t std_valid_width = (std_area.width / 2) / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("extended_offset", extended_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width", desc.width / unit_bytes));
    _shader_uv->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (std_valid_width, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (std_area.height / 2, 4) / 4;
    groups_size.z = 1;
    _shader_uv->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

FastMapYUV420::FastMapYUV420 (GLGeoMapHandler *mapper)
    : FastMap (mapper)
{
}

XCamReturn
FastMapYUV420::init_shaders ()
{
    _shader_y = create_shader (ShaderFastMapY, "fastmap_program_yuv420_y");
    _shader_uv = create_shader (ShaderFastMapUVYUV420, "fastmap_program_yuv420_uv");
    XCAM_ASSERT (_shader_y.ptr () && _shader_uv.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
FastMapYUV420::start_y (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf)
{
    const SmartPtr<GLBuffer> &coordx_y = _mapper->get_coordx_y ();
    const SmartPtr<GLBuffer> &coordy_y = _mapper->get_coordy_y ();

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0, YUV420PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 1, YUV420PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (coordx_y, 2));
    cmds.push_back (new GLCmdBindBufRange (coordy_y, 3));
    _shader_y->set_commands (cmds);

    return _shader_y->work (NULL);
};

XCamReturn
FastMapYUV420::start_uv (const SmartPtr<GLBuffer> &in_buf, const SmartPtr<GLBuffer> &out_buf)
{
    const SmartPtr<GLBuffer> &coordx_uv = _mapper->get_coordx_uv ();
    const SmartPtr<GLBuffer> &coordy_uv = _mapper->get_coordy_uv ();

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0, YUV420PlaneUIdx));
    cmds.push_back (new GLCmdBindBufRange (in_buf, 1, YUV420PlaneVIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 2, YUV420PlaneUIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 3, YUV420PlaneVIdx));
    cmds.push_back (new GLCmdBindBufRange (coordx_uv, 4));
    cmds.push_back (new GLCmdBindBufRange (coordy_uv, 5));
    _shader_uv->set_commands (cmds);

    return _shader_uv->work (NULL);
};

XCamReturn
FastMapYUV420::fix_uv_parameters (const VideoBufferInfo &in_info)
{
    const SmartPtr<GLBuffer> &coordx_uv = _mapper->get_coordx_uv ();
    const GLBufferDesc &desc = coordx_uv->get_buffer_desc ();
    const Rect &std_area = _mapper->get_std_area ();

    uint32_t width, height;
    _mapper->get_output_size (width, height);

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = (in_info.width / 2) / unit_bytes;
    uint32_t out_img_width = (width / 2) / unit_bytes;
    uint32_t extended_offset = (_mapper->get_extended_offset () / 2) / unit_bytes;
    uint32_t std_valid_width = (std_area.width / 2) / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("extended_offset", extended_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width", desc.width / unit_bytes));
    _shader_uv->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (std_valid_width, 2) / 2;
    groups_size.y = XCAM_ALIGN_UP (std_area.height / 2, 4) / 4;
    groups_size.z = 1;
    _shader_uv->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

} // GLGeoMapPriv

GLGeoMapHandler::GLGeoMapHandler (const char *name)
    : GLImageHandler (name)
    , _left_factor_x (0.0f)
    , _left_factor_y (0.0f)
    , _right_factor_x (0.0f)
    , _right_factor_y (0.0f)
    , _extended_offset (0)
    , _activate_fastmap (false)
    , _fastmap_activated (false)
{
    xcam_mem_clear (_lut_step);
}

GLGeoMapHandler::~GLGeoMapHandler ()
{
    terminate ();
}

bool
GLGeoMapHandler::set_lookup_table (
    const PointFloat2 *data, uint32_t width, uint32_t height)
{
    XCAM_FAIL_RETURN (
        ERROR, data && width && height, false,
        "gl-geomap set look up table failed, data ptr: %p, size: %dx%d",
        data, width, height);

    uint32_t lut_size = width * height * 2 * sizeof (float);
    SmartPtr<GLBuffer> buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, lut_size);
    XCAM_ASSERT (buf.ptr ());

    GLBufferDesc desc;
    desc.width = width;
    desc.height = height;
    desc.size = lut_size;
    buf->set_buffer_desc (desc);

    float *ptr = (float *) buf->map_range (0, lut_size, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, ptr, false, "gl-geomap map range failed");
    for (uint32_t i = 0; i < height; ++i) {
        float *ret = &ptr[i * width * 2];
        const PointFloat2 *line = &data[i * width];

        for (uint32_t j = 0; j < width; ++j) {
            ret[j * 2] = line[j].x;
            ret[j * 2 + 1] = line[j].y;
        }
    }
    buf->unmap ();

    _lut_buf = buf;

    return true;
}

void
GLGeoMapHandler::get_left_factors (float &x, float &y)
{
    x = _left_factor_x;
    y = _left_factor_y;
}

void
GLGeoMapHandler::get_right_factors (float &x, float &y)
{
    x = _right_factor_x;
    y = _right_factor_y;
}

bool
GLGeoMapHandler::set_std_area (const Rect &area)
{
    _std_area = area;
    return true;
}

const Rect &
GLGeoMapHandler::get_std_area () const
{
    return _std_area;
}

bool
GLGeoMapHandler::set_extended_offset (uint32_t offset)
{
    _extended_offset = offset;
    return true;
}

uint32_t
GLGeoMapHandler::get_extended_offset () const
{
   return _extended_offset;
}

bool
GLGeoMapHandler::set_lut_buf (const SmartPtr<GLBuffer> &buf)
{
    _lut_buf = buf;
    return true;
}

const SmartPtr<GLBuffer> &
GLGeoMapHandler::get_lut_buf () const
{
    if (!_lut_buf.ptr ()) {
        XCAM_LOG_ERROR ("gl-geomap lut buffer is empty, need set lookup table first");
    }

    return _lut_buf;
}

bool
GLGeoMapHandler::set_coordx_y (const SmartPtr<GLBuffer> &coordx_y)
{
    _coordx_y = coordx_y;
    return true;
}

const SmartPtr<GLBuffer> &
GLGeoMapHandler::get_coordx_y () const
{
    if (!_coordx_y.ptr ()) {
        XCAM_LOG_ERROR ("gl-geomap coordx buffer is empty");
    }

    return _coordx_y;
}

bool
GLGeoMapHandler::set_coordy_y (const SmartPtr<GLBuffer> &coordy_y)
{
    _coordy_y = coordy_y;
    return true;
}

const SmartPtr<GLBuffer> &
GLGeoMapHandler::get_coordy_y () const
{
    if (!_coordy_y.ptr ()) {
        XCAM_LOG_ERROR ("gl-geomap coordy buffer is empty");
    }

    return _coordy_y;
}

bool
GLGeoMapHandler::set_coordx_uv (const SmartPtr<GLBuffer> &coordx_uv)
{
    _coordx_uv = coordx_uv;
    return true;
}

const SmartPtr<GLBuffer> &
GLGeoMapHandler::get_coordx_uv () const
{
    if (!_coordx_uv.ptr ()) {
        XCAM_LOG_ERROR ("gl-geomap coordx UV buffer is empty");
    }

    return _coordx_uv;
}

bool
GLGeoMapHandler::set_coordy_uv (const SmartPtr<GLBuffer> &coordy_uv)
{
    _coordy_uv = coordy_uv;
    return true;
}

const SmartPtr<GLBuffer> &
GLGeoMapHandler::get_coordy_uv () const
{
    if (!_coordy_uv.ptr ()) {
        XCAM_LOG_ERROR ("gl-geomap coordy UV buffer is empty");
    }

    return _coordy_uv;
}

const float *
GLGeoMapHandler::get_lut_step () const
{
    return _lut_step;
}

static void
update_lut_step (
    float *lut_step,
    float left_factor_x, float left_factor_y,
    float right_factor_x, float right_factor_y)
{
    lut_step[0] = 1.0f / left_factor_x;
    lut_step[1] = 1.0f / left_factor_y;
    lut_step[2] = 1.0f / right_factor_x;
    lut_step[3] = 1.0f / right_factor_y;
}

bool
GLGeoMapHandler::init_factors ()
{
    XCAM_FAIL_RETURN (
        ERROR, _lut_buf.ptr (), false,
        "gl-geomap init factors failed, look up table is empty");

    const GLBufferDesc &lut_desc = _lut_buf->get_buffer_desc ();
    XCAM_FAIL_RETURN (
        ERROR, auto_calculate_factors (lut_desc.width, lut_desc.height), false,
        "gl-geomap auto calculate factors failed");

    get_factors (_left_factor_x, _left_factor_y);
    _right_factor_x = _left_factor_x;
    _right_factor_y = _left_factor_y;

    update_lut_step (_lut_step, _left_factor_x, _left_factor_y, _right_factor_x, _right_factor_y);

    return true;
}

bool
GLGeoMapHandler::update_factors (
    float left_factor_x, float left_factor_y, float right_factor_x, float right_factor_y)
{
    XCAM_FAIL_RETURN (
        ERROR,
        !XCAM_DOUBLE_EQUAL_AROUND (left_factor_x, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (left_factor_y, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (right_factor_x, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (right_factor_y, 0.0f), false,
        "gl-geomap invalid factors, left factors: %f, %f, right factors: %f, %f",
        left_factor_x, left_factor_y, right_factor_x, right_factor_y);

    _left_factor_x = (left_factor_x + right_factor_x) / 2.0f;
    _left_factor_y = (left_factor_y + right_factor_y) / 2.0f;
    _right_factor_x = _left_factor_x;
    _right_factor_y = _left_factor_y;

    update_lut_step (_lut_step, _left_factor_x, _left_factor_y, _right_factor_x, _right_factor_y);

    return true;
}

void
GLGeoMapHandler::activate_fastmap ()
{
    _activate_fastmap = true;
}

static XCamReturn
set_output_video_info (
    const SmartPtr<GLGeoMapHandler> &handler, const SmartPtr<ImageHandler::Parameters> &param)
{
    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();

    uint32_t width, height;
    handler->get_output_size (width, height);

    VideoBufferInfo out_info;
    out_info.init (
        in_info.format, width, height,
        XCAM_ALIGN_UP (width, XCAM_GL_GEOMAP_ALIGN_X),
        XCAM_ALIGN_UP (height, XCAM_GL_GEOMAP_ALIGN_Y));
    handler->set_out_video_info (out_info);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::ensure_default_params ()
{
    uint32_t width, height;
    get_output_size (width, height);
    XCAM_FAIL_RETURN (
        ERROR, width && height, XCAM_RETURN_ERROR_PARAM,
        "gl-geomap invalid output size %dx%d", width, height);

    if (_std_area.width == 0 || _std_area.height == 0) {
        Rect area = Rect (0, 0, width, height);
        set_std_area (area);
    }

    uint32_t std_width, std_height;
    get_std_output_size (std_width, std_height);
    if (std_width == 0 || std_height == 0) {
        set_std_output_size (width, height);
    }

    if (_right_factor_x == 0.0f || _right_factor_y == 0.0f ||
        _left_factor_x == 0.0f  || _left_factor_y == 0.0f ) {
        init_factors ();
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (need_allocator ()) {
        ret = set_output_video_info (this, param);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "gl-geomap set output video info failed");
    }

    ret = ensure_default_params ();
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-geomap ensure default paramaters failed");

    const VideoBufferInfo &info = param->in_buf->get_video_info ();
    SmartPtr<GLGeoMapPriv::ComMap> commapper;
    if (info.format == V4L2_PIX_FMT_NV12) {
        commapper = new GLGeoMapPriv::ComMapNV12 (this);
    } else {
        commapper = new GLGeoMapPriv::ComMapYUV420 (this);
    }
    XCAM_ASSERT (commapper.ptr ());

    ret = commapper->configure_resource (param);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-geomap configure resource failed");

    _commapper = commapper;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());

    if (!_fastmap_activated) {
        if (_activate_fastmap) {
            _commapper->prepare_dump_coords ();
        }

        _commapper->start (param);
    }

    if (_activate_fastmap && !_fastmap_activated) {
        GLSync::finish ();

        _lut_buf.release ();
        _commapper.release ();

        const VideoBufferInfo &info = param->in_buf->get_video_info ();
        SmartPtr<GLGeoMapPriv::FastMap> fastmapper;
        if (info.format == V4L2_PIX_FMT_NV12) {
            fastmapper = new GLGeoMapPriv::FastMapNV12 (this);
        } else {
            fastmapper = new GLGeoMapPriv::FastMapYUV420 (this);
        }
        XCAM_ASSERT (fastmapper.ptr ());

        fastmapper->configure_resource (param);
        _fastmapper = fastmapper;

        _fastmap_activated = true;
        _activate_fastmap = false;
    }

    if (_fastmap_activated) {
        _fastmapper->start (param);
    }

    param->in_buf.release ();

    return XCAM_RETURN_NO_ERROR;
};

XCamReturn
GLGeoMapHandler::remap (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret, "gl-geomap remap failed");

    GLSync::flush ();
    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

XCamReturn
GLGeoMapHandler::terminate ()
{
    if (_lut_buf.ptr ()) {
        _lut_buf.release ();
    }

    if (_coordx_y.ptr ()) {
        _coordx_y.release ();
    }
    if (_coordy_y.ptr ()) {
        _coordy_y.release ();
    }
    if (_coordx_uv.ptr ()) {
        _coordx_uv.release ();
    }
    if (_coordy_uv.ptr ()) {
        _coordy_uv.release ();
    }

    if (_commapper.ptr ()) {
        _commapper.release ();
    }
    if (_fastmapper.ptr ()) {
        _fastmapper.release ();
    }

    return GLImageHandler::terminate ();
}

bool
GLDualConstGeoMapHandler::update_factors (
    float left_factor_x, float left_factor_y, float right_factor_x, float right_factor_y)
{
    XCAM_FAIL_RETURN (
        ERROR,
        !XCAM_DOUBLE_EQUAL_AROUND (left_factor_x, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (left_factor_y, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (right_factor_x, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (right_factor_y, 0.0f), false,
        "gl-geomap update lut step failed, left factors: %f %f, right factors: %f %f",
        left_factor_x, left_factor_y, right_factor_x, right_factor_y);

    _left_factor_x = left_factor_x;
    _left_factor_y = left_factor_y;
    _right_factor_x = right_factor_x;
    _right_factor_y = right_factor_y;

    update_lut_step (_lut_step, _left_factor_x, _left_factor_y, _right_factor_x, _right_factor_y);

    return true;
}

SmartPtr<GLImageHandler> create_gl_geo_mapper ()
{
    SmartPtr<GLImageHandler> mapper = new GLGeoMapHandler ();
    XCAM_ASSERT (mapper.ptr ());

    return mapper;
}

SmartPtr<GeoMapper>
GeoMapper::create_gl_geo_mapper ()
{
    SmartPtr<GLImageHandler> handler = XCam::create_gl_geo_mapper ();
    return handler.dynamic_cast_ptr<GeoMapper> ();
}

}
