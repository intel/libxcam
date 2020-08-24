/*
 * gl_copy_handler.cpp - gl copy handler implementation
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

#include "gl_copy_handler.h"
#include "gl_utils.h"

#define INVALID_INDEX (uint32_t)(-1)

namespace XCam {

const GLShaderInfo shader_info = {
    GL_COMPUTE_SHADER,
    "shader_copy",
#include "shader_copy.comp.slx"
    , 0
};

GLCopyHandler::GLCopyHandler (const char *name)
    : GLImageHandler (name)
    , _index (INVALID_INDEX)
{
}

XCamReturn
GLCopyHandler::copy (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-copy execute copy failed");

    _copy_shader->finish ();
    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

bool
GLCopyHandler::set_copy_area (uint32_t idx, const Rect &in_area, const Rect &out_area)
{
    XCAM_FAIL_RETURN (
        ERROR,
        idx != INVALID_INDEX &&
        in_area.width == out_area.width && in_area.height == out_area.height,
        false,
        "gl-copy set copy area failed, idx: %d, input size: %dx%d, output size: %dx%d",
        idx, in_area.width, in_area.height, out_area.width, out_area.height);

    _index = idx;
    _in_area = in_area;
    _out_area = out_area;

    XCAM_LOG_DEBUG (
        "gl-copy set copy area, idx: %d, input area: %d, %d, %d, %d, output area: %d, %d, %d, %d",
        idx,
        in_area.pos_x, in_area.pos_y, in_area.width, in_area.height,
        out_area.pos_x, out_area.pos_y, out_area.width, out_area.height);

    return true;
}

XCamReturn
GLCopyHandler::fix_parameters (const SmartPtr<Parameters> &param)
{
    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &out_info =
        param->out_buf.ptr () ? param->out_buf->get_video_info () : get_out_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, out_info.width > 0, XCAM_RETURN_ERROR_PARAM,
        "gl-copy invalid output width: %d", out_info.width);

    const size_t unit_bytes = sizeof (uint32_t) * 4;
    uint32_t in_img_width = in_info.aligned_width / unit_bytes;
    uint32_t in_x_offset = _in_area.pos_x / unit_bytes;
    uint32_t out_img_width = out_info.aligned_width / unit_bytes;
    uint32_t out_x_offset = _out_area.pos_x / unit_bytes;
    uint32_t copy_w = _in_area.width / unit_bytes;
    uint32_t copy_h = _in_area.height / 2 * 3;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_x_offset", in_x_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_x_offset", out_x_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("copy_width", copy_w));
    _copy_shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (copy_w, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (copy_h, 8) / 8;
    groups_size.z = 1;
    _copy_shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLCopyHandler::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        _index != INVALID_INDEX &&
        _in_area.width && _in_area.height && _out_area.width && _out_area.height,
        XCAM_RETURN_ERROR_PARAM,
        "gl-copy invalid copy area, index: %d, in size: %dx%d, out size: %dx%d",
        _index, _in_area.width, _in_area.height, _out_area.width, _out_area.height);

    SmartPtr<GLImageShader> shader = new GLImageShader (shader_info.name);
    XCAM_ASSERT (shader.ptr ());
    _copy_shader = shader;

    XCamReturn ret = _copy_shader->create_compute_program (shader_info);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-copy create %s program failed", shader_info.name);

    fix_parameters (param);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLCopyHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    SmartPtr<GLBuffer> in_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in_buf, 0));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 1));
    _copy_shader->set_commands (cmds);

    return _copy_shader->work (NULL);
};

XCamReturn
GLCopyHandler::terminate ()
{
    if (_copy_shader.ptr ()) {
        _copy_shader.release ();
    }

    return GLImageHandler::terminate ();
}

}
