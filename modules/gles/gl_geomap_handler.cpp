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

enum ShaderID {
    ShaderComMap = 0,    // common mapping
    ShaderFastMap        // fast mapping
};

static const GLShaderInfo shaders_info[] = {
    {
        GL_COMPUTE_SHADER,
        "shader_geomap",
#include "shader_geomap.comp.slx"
    , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_geomap_fastmap",
#include "shader_geomap_fastmap.comp.slx"
    , 0
    },
};

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
    XCAM_FAIL_RETURN (
        ERROR, _lut_buf.ptr (), NULL,
        "gl-geomap lut buffer is empty, need set lookup table first");

    return _lut_buf;
}

const SmartPtr<GLBuffer> &
GLGeoMapHandler::get_coordx_buf () const
{
    XCAM_FAIL_RETURN (
        ERROR, _coordx_buf.ptr (), NULL,
        "gl-geomap coordx buffer is empty");

    return _coordx_buf;
}

const SmartPtr<GLBuffer> &
GLGeoMapHandler::get_coordy_buf () const
{
    XCAM_FAIL_RETURN (
        ERROR, _coordy_buf.ptr (), NULL,
        "gl-geomap coordy buffer is empty");

    return _coordy_buf;
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

XCamReturn
GLGeoMapHandler::fix_parameters (const SmartPtr<Parameters> &param)
{
    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    const GLBufferDesc &lut_desc = _lut_buf->get_buffer_desc ();

    float factor_x, factor_y;
    get_factors (factor_x, factor_y);
    float lut_std_step[2] = {1.0f / factor_x, 1.0f / factor_y};

    uint32_t width, height, std_width, std_height;
    get_output_size (width, height);
    get_std_output_size (std_width, std_height);

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.width / unit_bytes;
    uint32_t out_img_width = width / unit_bytes;
    uint32_t extended_offset = _extended_offset / unit_bytes;

    std_width /= unit_bytes;
    uint32_t std_offset = _std_area.pos_x / unit_bytes;
    uint32_t std_valid_width = _std_area.width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", in_info.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("extended_offset", extended_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_width", std_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_height", _std_area.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("std_offset", std_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_width", lut_desc.width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_height", lut_desc.height));
    cmds.push_back (new GLCmdUniformTVect<float, 2> ("lut_std_step", lut_std_step));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("dump_coords", 0));
    _geomap_shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (std_valid_width, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (_std_area.height, 8) / 8;
    groups_size.z = 1;
    _geomap_shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
set_output_video_info (
    const SmartPtr<GLGeoMapHandler> &handler, const SmartPtr<ImageHandler::Parameters> &param)
{
    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "gl-geomap only support NV12 format, but input format is %s",
        xcam_fourcc_to_string (in_info.format));

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

    SmartPtr<GLImageShader> shader = new GLImageShader (shaders_info[ShaderComMap].name);
    XCAM_ASSERT (shader.ptr ());

    ret = shader->create_compute_program (shaders_info[ShaderComMap], "geomap_program");
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-geomap create compute program for common mapping failed");
    _geomap_shader = shader;

    fix_parameters (param);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::prepare_dump_coords (GLCmdList &cmds)
{
    GLBufferDesc desc;
    desc.width = _std_area.width;
    desc.height = _std_area.height;
    desc.size = desc.width * desc.height * sizeof (float);

    SmartPtr<GLBuffer> coordx_buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    SmartPtr<GLBuffer> coordy_buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    XCAM_ASSERT (coordx_buf.ptr () && coordy_buf.ptr ());

    coordx_buf->set_buffer_desc (desc);
    coordy_buf->set_buffer_desc (desc);
    _coordx_buf = coordx_buf;
    _coordy_buf = coordy_buf;

    cmds.push_back (new GLCmdUniformT<uint32_t> ("dump_coords", 1));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width", desc.width / sizeof (uint32_t)));
    cmds.push_back (new GLCmdBindBufBase (_coordx_buf, 5));
    cmds.push_back (new GLCmdBindBufBase (_coordy_buf, 6));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::switch_to_fastmap (const SmartPtr<ImageHandler::Parameters> &param)
{
    if (_fastmap_activated)
        return XCAM_RETURN_NO_ERROR;

    _lut_buf.release ();
    _geomap_shader.release ();

    SmartPtr<GLImageShader> shader = new GLImageShader (shaders_info[ShaderFastMap].name);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shaders_info[ShaderFastMap], "fastmap_program");
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-geomap create compute program for fast mapping failed");
    _geomap_shader = shader;

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    const GLBufferDesc &desc = _coordx_buf->get_buffer_desc ();

    uint32_t width, height;
    get_output_size (width, height);

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.width / unit_bytes;
    uint32_t out_img_width = width / unit_bytes;
    uint32_t extended_offset = _extended_offset / unit_bytes;
    uint32_t std_valid_width = _std_area.width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", in_info.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("extended_offset", extended_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width", desc.width / unit_bytes));
    _geomap_shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (std_valid_width, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (_std_area.height, 8) / 8;
    groups_size.z = 1;
    _geomap_shader->set_groups_size (groups_size);

    _fastmap_activated = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());

    SmartPtr<GLBuffer> in_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (in_buf, 1, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 2, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 3, NV12PlaneUVIdx));
    if (_fastmap_activated) {
        cmds.push_back (new GLCmdBindBufRange (_coordx_buf, 4));
        cmds.push_back (new GLCmdBindBufRange (_coordy_buf, 5));
    } else {
        cmds.push_back (new GLCmdBindBufBase (_lut_buf, 4));
        cmds.push_back (new GLCmdUniformTVect<float, 4> ("lut_step", _lut_step));
    }
    if (_activate_fastmap && !_fastmap_activated) {
        prepare_dump_coords (cmds);
    }

    _geomap_shader->set_commands (cmds);
    _geomap_shader->work (NULL);

    if (_activate_fastmap && !_fastmap_activated) {
        GLSync::finish ();
        switch_to_fastmap (param);
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
    if (_geomap_shader.ptr ()) {
        _geomap_shader.release ();
    }

    if (_lut_buf.ptr ()) {
        _lut_buf.release ();
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
