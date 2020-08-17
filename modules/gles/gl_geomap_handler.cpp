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

#define XCAM_GL_GEOMAP_ALIGN_X 4
#define XCAM_GL_GEOMAP_ALIGN_Y 2

namespace XCam {

const GLShaderInfo shader_info = {
    GL_COMPUTE_SHADER,
    "shader_geomap",
#include "shader_geomap.comp.slx"
    , 0
};

GLGeoMapHandler::GLGeoMapHandler (const char *name)
    : GLImageHandler (name)
    , _left_factor_x (0.0f)
    , _left_factor_y (0.0f)
    , _right_factor_x (0.0f)
    , _right_factor_y (0.0f)
{
    xcam_mem_clear (_lut_step);
}

bool
GLGeoMapHandler::set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height)
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
    float factor_x, factor_y;
    get_factors (factor_x, factor_y);

    if (!XCAM_DOUBLE_EQUAL_AROUND (factor_x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factor_y, 0.0f))
        return true;

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

XCamReturn
GLGeoMapHandler::fix_parameters (
    const VideoBufferInfo &in_info, const VideoBufferInfo &out_info)
{
    const GLBufferDesc &lut_desc = _lut_buf->get_buffer_desc ();

    float factor_x, factor_y;
    get_factors (factor_x, factor_y);
    float lut_std_step[2] = {1.0f / factor_x, 1.0f / factor_y};

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = in_info.width / unit_bytes;
    uint32_t out_img_width = out_info.width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", in_info.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_height", out_info.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_width", lut_desc.width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_height", lut_desc.height));
    cmds.push_back (new GLCmdUniformTVect<float, 2> ("lut_std_step", lut_std_step));
    _geomap_shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (out_img_width, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (out_info.height, 16) / 16;
    groups_size.z = 1;
    _geomap_shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _lut_buf.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-geomap configure resource failed, look up table is empty");

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "gl-geomap only support NV12 format, but input format is %s",
        xcam_fourcc_to_string (in_info.format));

    uint32_t width, height;
    get_output_size (width, height);
    VideoBufferInfo out_info;
    out_info.init (
        in_info.format, width, height,
        XCAM_ALIGN_UP (width, XCAM_GL_GEOMAP_ALIGN_X),
        XCAM_ALIGN_UP (height, XCAM_GL_GEOMAP_ALIGN_Y));
    set_out_video_info (out_info);

    SmartPtr<GLImageShader> shader = new GLImageShader (shader_info.name);
    XCAM_ASSERT (shader.ptr ());
    XCamReturn ret = shader->create_compute_program (shader_info, "geomap_program");
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, XCAM_RETURN_ERROR_MEM,
        "gl-geomap create compute program failed");
    _geomap_shader = shader;

    init_factors ();
    fix_parameters (in_info, out_info);

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
    cmds.push_back (new GLCmdBindBufBase (_lut_buf, 4));
    cmds.push_back (new GLCmdUniformTVect<float, 4> ("lut_step", _lut_step));
    _geomap_shader->set_commands (cmds);

    param->in_buf.release ();

    return _geomap_shader->work (NULL);
};

XCamReturn
GLGeoMapHandler::remap (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret, "gl-geomap remap failed");

    _geomap_shader->finish ();
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
        "gl-geomap update lut step failed, left factors: %f, %f, right factors: %f, %f",
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
