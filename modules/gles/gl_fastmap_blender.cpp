/*
 * gl_fastmap_blender.cpp - gl fastmap blender implementation
 *
 *  Copyright (c) 2020 Intel Corporation
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

#include "xcam_utils.h"
#include "gl_utils.h"
#include "gl_fastmap_blender.h"

namespace XCam {

static const GLShaderInfo shader_info = {
    GL_COMPUTE_SHADER,
    "shader_fastmap_blend",
#include "shader_fastmap_blend.comp.slx"
    , 0
};

GLFastmapBlender::GLFastmapBlender (const char *name)
    : GLImageHandler (name)
{
}

GLFastmapBlender::~GLFastmapBlender ()
{
    terminate ();
}

XCamReturn
GLFastmapBlender::terminate ()
{
    if (_left_coordx.ptr ()) {
        _left_coordx.release ();
    }
    if (_left_coordy.ptr ()) {
        _left_coordy.release ();
    }
    if (_right_coordx.ptr ()) {
        _right_coordx.release ();
    }
    if (_right_coordy.ptr ()) {
        _right_coordy.release ();
    }
    if (_mask.ptr ()) {
        _mask.release ();
    }

    _shader.release ();

    return GLImageHandler::terminate ();
}

bool
GLFastmapBlender::set_fastmappers (
    const SmartPtr<GLGeoMapHandler> &left_mapper, const SmartPtr<GLGeoMapHandler> &right_mapper)
{
    _left_mapper = left_mapper;
    _right_mapper = right_mapper;

    return true;
}

bool
GLFastmapBlender::set_blender (const SmartPtr<GLBlender> &blender)
{
    _blender = blender;

    return true;
}

XCamReturn
GLFastmapBlender::init_mask (uint32_t width)
{
    XCAM_ASSERT (width);

    uint32_t buf_size = width * sizeof (uint8_t);
    SmartPtr<GLBuffer> buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, buf_size);
    XCAM_ASSERT (buf.ptr ());

    GLBufferDesc desc;
    desc.width = width;
    desc.height = 1;
    desc.size = buf_size;
    buf->set_buffer_desc (desc);

    std::vector<float> gauss_table;
    uint32_t quater = desc.width / 4;

    get_gauss_table (quater, (quater + 1) / 4.0f, gauss_table, false);
    for (uint32_t i = 0; i < gauss_table.size (); ++i) {
        float value = ((i < quater) ? (128.0f * (2.0f - gauss_table[i])) : (128.0f * gauss_table[i]));
        value = XCAM_CLAMP (value, 0.0f, 255.0f);
        gauss_table[i] = value;
    }

    uint8_t *mask_ptr = (uint8_t *) buf->map_range (0, buf_size, GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, mask_ptr, XCAM_RETURN_ERROR_PARAM, "map range failed");

    uint32_t gauss_start_pos = (desc.width - gauss_table.size ()) / 2;
    uint32_t idx = 0;
    for (idx = 0; idx < gauss_start_pos; ++idx) {
        mask_ptr[idx] = 255;
    }
    for (uint32_t i = 0; i < gauss_table.size (); ++idx, ++i) {
        mask_ptr[idx] = (uint8_t) gauss_table[i];
    }
    for (; idx < desc.width; ++idx) {
        mask_ptr[idx] = 0;
    }
    buf->unmap ();

    _mask = buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLFastmapBlender::transfer_buffers ()
{
    _left_coordx = _left_mapper->get_coordx_buf ();
    _left_coordy = _left_mapper->get_coordy_buf ();
    XCAM_FAIL_RETURN (
        ERROR, _left_coordx.ptr () && _left_coordy.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get left coordinate buffers");

    _right_coordx = _right_mapper->get_coordx_buf ();
    _right_coordy = _right_mapper->get_coordy_buf ();
    XCAM_ASSERT (_right_coordx.ptr () && _right_coordy.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _right_coordx.ptr () && _right_coordy.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get right coordinate buffers");

    _mask = _blender->get_layer0_mask ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLFastmapBlender::fix_parameters (const SmartPtr<Parameters> &base)
{
    SmartPtr<GLBlender::BlenderParam> param = base.dynamic_cast_ptr<GLBlender::BlenderParam> ();
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->in1_buf.ptr () && param->out_buf.ptr ());

    const Rect &blend_area = _blender->get_merge_window ();
    if (!_mask.ptr ()) {
        init_mask (blend_area.width);
    }

    const VideoBufferInfo &left_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &right_info = param->in1_buf->get_video_info ();
    const VideoBufferInfo &out_info = param->out_buf->get_video_info ();
    const GLBufferDesc &left_coord_desc = _left_coordx->get_buffer_desc ();
    const GLBufferDesc &right_coord_desc = _right_coordx->get_buffer_desc ();

    const size_t unit_bytes = sizeof (uint32_t);

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width0", left_info.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width1", right_info.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", left_info.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_info.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_offset_x", blend_area.pos_x / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width0", left_coord_desc.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width1", right_coord_desc.width / unit_bytes));
    _shader->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (blend_area.width / unit_bytes, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (blend_area.height, 8) / 8;
    groups_size.z = 1;
    _shader->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLFastmapBlender::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr ());

    SmartPtr<GLImageShader> shader = new GLImageShader (shader_info.name);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shader_info, "fastmap_blender_program");
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender create fastmap-blender compute program failed");
    _shader = shader;

    ret = transfer_buffers ();
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender transfer buffers failed");

    ret = fix_parameters (param);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender fix parameters failed");

    _left_mapper->terminate ();
    _left_mapper.release ();
    _right_mapper->terminate ();
    _right_mapper.release ();
    _blender->terminate ();
    _blender.release ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLFastmapBlender::start_work (const SmartPtr<ImageHandler::Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());

    SmartPtr<GLBlender::BlenderParam> param = base.dynamic_cast_ptr<GLBlender::BlenderParam> ();
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->in1_buf.ptr () && param->out_buf.ptr ());

    SmartPtr<GLBuffer> in0_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> in1_buf = get_glbuffer (param->in1_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 0, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 1, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_left_coordx, 2));
    cmds.push_back (new GLCmdBindBufRange (_left_coordy, 3));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 4, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 5, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_right_coordx, 6));
    cmds.push_back (new GLCmdBindBufRange (_right_coordy, 7));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 8, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 9, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_mask, 10));
    _shader->set_commands (cmds);

    return _shader->work (NULL);
};

}
