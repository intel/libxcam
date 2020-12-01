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

namespace GLFastmapBlendPriv {

class Impl
{
public:
    Impl () {}
    virtual ~Impl ();

    virtual XCamReturn start (const SmartPtr<GLBlender::BlenderParam> &param) = 0;
    virtual XCamReturn configure_resource (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param,
        const SmartPtr<GLGeoMapHandler> &left_mapper, const SmartPtr<GLGeoMapHandler> &right_mapper) = 0;

protected:
    XCamReturn init_mask_y (uint32_t width);

protected:
    SmartPtr<GLBuffer>    _mask_y;
    SmartPtr<GLBuffer>    _left_coordx_y;
    SmartPtr<GLBuffer>    _left_coordy_y;
    SmartPtr<GLBuffer>    _right_coordx_y;
    SmartPtr<GLBuffer>    _right_coordy_y;
};

class ImplNV12
    : public Impl
{
public:
    ImplNV12 () {}
    virtual ~ImplNV12 ();

    virtual XCamReturn start (const SmartPtr<GLBlender::BlenderParam> &param);
    virtual XCamReturn configure_resource (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param,
        const SmartPtr<GLGeoMapHandler> &left_mapper, const SmartPtr<GLGeoMapHandler> &right_mapper);

private:
    XCamReturn transfer_buffers (
        const SmartPtr<GLBlender> &blender,
        const SmartPtr<GLGeoMapHandler> &left_mapper,
        const SmartPtr<GLGeoMapHandler> &right_mapper);
    XCamReturn fix_parameters (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param);

private:
    SmartPtr<GLImageShader>    _shader;
};

Impl::~Impl ()
{
    _mask_y.release ();
    _left_coordx_y.release ();
    _left_coordy_y.release ();
    _right_coordx_y.release ();
    _right_coordy_y.release ();
}

XCamReturn
Impl::init_mask_y (uint32_t width)
{
    XCAM_ASSERT (width);

    GLBufferDesc desc;
    desc.width = width;
    desc.height = 1;
    desc.size = width * sizeof (uint8_t);

    SmartPtr<GLBuffer> mask = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc.size);
    XCAM_ASSERT (mask.ptr ());
    mask->set_buffer_desc (desc);

    std::vector<float> gauss_table;
    uint32_t quater = desc.width / 4;

    get_gauss_table (quater, (quater + 1) / 4.0f, gauss_table, false);
    for (uint32_t i = 0; i < gauss_table.size (); ++i) {
        float value = ((i < quater) ? (128.0f * (2.0f - gauss_table[i])) : (128.0f * gauss_table[i]));
        value = XCAM_CLAMP (value, 0.0f, 255.0f);
        gauss_table[i] = value;
    }

    uint8_t *mask_ptr = (uint8_t *) mask->map_range (0, desc.size, GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, mask_ptr, XCAM_RETURN_ERROR_PARAM, "gl-fastmap_blender map range failed");

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
    mask->unmap ();

    _mask_y = mask;

    return XCAM_RETURN_NO_ERROR;
}

ImplNV12::~ImplNV12 ()
{
    _shader.release ();
}

XCamReturn
ImplNV12::start (const SmartPtr<GLBlender::BlenderParam> &param)
{
    SmartPtr<GLBuffer> in0_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> in1_buf = get_glbuffer (param->in1_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);

    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 0, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 1, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_left_coordx_y, 2));
    cmds.push_back (new GLCmdBindBufRange (_left_coordy_y, 3));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 4, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 5, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_right_coordx_y, 6));
    cmds.push_back (new GLCmdBindBufRange (_right_coordy_y, 7));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 8, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 9, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_mask_y, 10));
    _shader->set_commands (cmds);

    return _shader->work (NULL);
};

XCamReturn
ImplNV12::configure_resource (
    const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param,
    const SmartPtr<GLGeoMapHandler> &left_mapper, const SmartPtr<GLGeoMapHandler> &right_mapper)
{
    SmartPtr<GLImageShader> shader = new GLImageShader (shader_info.name);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shader_info, "fastmap_blend_program_nv12");
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender create compute program for %s failed", shader_info.name);
    _shader = shader;

    ret = transfer_buffers (blender, left_mapper, right_mapper);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender transfer buffers failed");

    ret = fix_parameters (blender, param);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender fix parameters failed");

    if (!_mask_y.ptr ()) {
        const Rect &area = blender->get_merge_window ();
        ret = init_mask_y (area.width);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "gl-fastmap_blender init mask failed");
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImplNV12::transfer_buffers (
    const SmartPtr<GLBlender> &blender,
    const SmartPtr<GLGeoMapHandler> &left_mapper,
    const SmartPtr<GLGeoMapHandler> &right_mapper)
{
    _left_coordx_y = left_mapper->get_coordx_y ();
    _left_coordy_y = left_mapper->get_coordy_y ();
    XCAM_FAIL_RETURN (
        ERROR, _left_coordx_y.ptr () && _left_coordy_y.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get left coordinate buffers");

    _right_coordx_y = right_mapper->get_coordx_y ();
    _right_coordy_y = right_mapper->get_coordy_y ();
    XCAM_ASSERT (_right_coordx_y.ptr () && _right_coordy_y.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _right_coordx_y.ptr () && _right_coordy_y.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get right coordinate buffers");

    _mask_y = blender->get_layer0_mask ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImplNV12::fix_parameters (
    const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param)
{
    const VideoBufferInfo &left_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &right_info = param->in1_buf->get_video_info ();
    const VideoBufferInfo &out_info = param->out_buf->get_video_info ();
    const GLBufferDesc &left_coord_desc = _left_coordx_y->get_buffer_desc ();
    const GLBufferDesc &right_coord_desc = _right_coordx_y->get_buffer_desc ();
    const Rect &blend_area = blender->get_merge_window ();

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

} // GLFastmapBlendPriv

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
GLFastmapBlender::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr ());

    SmartPtr<GLBlender::BlenderParam> blend_param = param.dynamic_cast_ptr<GLBlender::BlenderParam> ();
    XCAM_ASSERT (blend_param->in_buf.ptr () && blend_param->in1_buf.ptr () && blend_param->out_buf.ptr ());

    SmartPtr<GLFastmapBlendPriv::Impl> impl = new GLFastmapBlendPriv::ImplNV12 ();
    XCAM_ASSERT (impl.ptr ());

    XCamReturn ret = impl->configure_resource (_blender, blend_param, _left_mapper, _right_mapper);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender configure resource failed");

    _impl = impl;

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

   return _impl->start (param);
}

}
