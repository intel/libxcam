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

enum ShaderID {
    ShaderFastmapBlendY = 0,
    ShaderFastmapBlendUVNV12,
    ShaderFastmapBlendUVYUV420
};

static const GLShaderInfo shaders_info[] = {
    {
        GL_COMPUTE_SHADER,
        "shader_fastmap_blend_y",
#include "shader_fastmap_blend_y.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_fastmap_blend_uv_nv12",
#include "shader_fastmap_blend_uv_nv12.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_fastmap_blend_uv_yuv420",
#include "shader_fastmap_blend_uv_yuv420.comp.slx"
        , 0
    }
};

namespace GLFastmapBlendPriv {

class Impl
{
public:
    Impl () {}
    virtual ~Impl ();

    XCamReturn start (const SmartPtr<GLBlender::BlenderParam> &param);
    XCamReturn configure_resource (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param,
        const SmartPtr<GLGeoMapHandler> &left_mapper, const SmartPtr<GLGeoMapHandler> &right_mapper);

protected:
    virtual XCamReturn init_shaders () = 0;
    virtual XCamReturn start_y (
        const SmartPtr<GLBuffer> &in0_buf,
        const SmartPtr<GLBuffer> &in1_buf,
        const SmartPtr<GLBuffer> &out_buf) = 0;
    virtual XCamReturn start_uv (
        const SmartPtr<GLBuffer> &in0_buf,
        const SmartPtr<GLBuffer> &in1_buf,
        const SmartPtr<GLBuffer> &out_buf) = 0;
    virtual XCamReturn fix_y_parameters (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param) = 0;
    virtual XCamReturn fix_uv_parameters (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param) = 0;

    XCamReturn init_mask_y (uint32_t width);
    XCamReturn init_mask_uv ();
    XCamReturn transfer_buffers (
        const SmartPtr<GLBlender> &blender,
        const SmartPtr<GLGeoMapHandler> &left_mapper,
        const SmartPtr<GLGeoMapHandler> &right_mapper);

protected:
    SmartPtr<GLImageShader>    _shader_y;
    SmartPtr<GLImageShader>    _shader_uv;

    SmartPtr<GLBuffer>         _mask_y;
    SmartPtr<GLBuffer>         _mask_uv;

    SmartPtr<GLBuffer>         _left_coordx_y;
    SmartPtr<GLBuffer>         _left_coordy_y;
    SmartPtr<GLBuffer>         _right_coordx_y;
    SmartPtr<GLBuffer>         _right_coordy_y;
    SmartPtr<GLBuffer>         _left_coordx_uv;
    SmartPtr<GLBuffer>         _left_coordy_uv;
    SmartPtr<GLBuffer>         _right_coordx_uv;
    SmartPtr<GLBuffer>         _right_coordy_uv;
};

class ImplNV12
    : public Impl
{
public:
    ImplNV12 () {}
    virtual ~ImplNV12 () {}

private:
    virtual XCamReturn init_shaders ();
    virtual XCamReturn start_y (
        const SmartPtr<GLBuffer> &in0_buf,
        const SmartPtr<GLBuffer> &in1_buf,
        const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn start_uv (
        const SmartPtr<GLBuffer> &in0_buf,
        const SmartPtr<GLBuffer> &in1_buf,
        const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn fix_y_parameters (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param);
    virtual XCamReturn fix_uv_parameters (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param);
};

class ImplYUV420
    : public Impl
{
public:
    ImplYUV420 () {}
    virtual ~ImplYUV420 () {}

private:
    virtual XCamReturn init_shaders ();
    virtual XCamReturn start_y (
        const SmartPtr<GLBuffer> &in0_buf,
        const SmartPtr<GLBuffer> &in1_buf,
        const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn start_uv (
        const SmartPtr<GLBuffer> &in0_buf,
        const SmartPtr<GLBuffer> &in1_buf,
        const SmartPtr<GLBuffer> &out_buf);
    virtual XCamReturn fix_y_parameters (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param);
    virtual XCamReturn fix_uv_parameters (
        const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param);
};

Impl::~Impl ()
{
    _shader_y.release ();
    _shader_uv.release ();

    _mask_y.release ();
    _mask_uv.release ();

    _left_coordx_y.release ();
    _left_coordy_y.release ();
    _right_coordx_y.release ();
    _right_coordy_y.release ();
    _left_coordx_uv.release ();
    _left_coordy_uv.release ();
    _right_coordx_uv.release ();
    _right_coordy_uv.release ();
}

XCamReturn
Impl::start (const SmartPtr<GLBlender::BlenderParam> &param)
{
    SmartPtr<GLBuffer> in0_buf = get_glbuffer (param->in_buf);
    SmartPtr<GLBuffer> in1_buf = get_glbuffer (param->in1_buf);
    SmartPtr<GLBuffer> out_buf = get_glbuffer (param->out_buf);

    XCamReturn ret = start_y (in0_buf, in1_buf, out_buf);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret, "gl-fastmap_blender start Y failed");

    ret = start_uv (in0_buf, in1_buf, out_buf);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret, "gl-fastmap_blender start UV failed");

    return XCAM_RETURN_NO_ERROR;
};

XCamReturn
Impl::configure_resource (
    const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param,
    const SmartPtr<GLGeoMapHandler> &left_mapper, const SmartPtr<GLGeoMapHandler> &right_mapper)
{
    init_shaders ();

    XCamReturn ret = transfer_buffers (blender, left_mapper, right_mapper);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-fastmap_blender transfer buffers failed");

    fix_y_parameters (blender, param);
    fix_uv_parameters (blender, param);

    if (!_mask_y.ptr ()) {
        const Rect &area = blender->get_merge_window ();
        ret = init_mask_y (area.width);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "gl-fastmap_blender init mask_y failed");
    }

    if (!_mask_uv.ptr ()) {
        ret = init_mask_uv ();
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "gl-fastmap_blender init mask_uv failed");
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Impl::transfer_buffers (
    const SmartPtr<GLBlender> &blender,
    const SmartPtr<GLGeoMapHandler> &left_mapper,
    const SmartPtr<GLGeoMapHandler> &right_mapper)
{
    _left_coordx_y = left_mapper->get_coordx_y ();
    _left_coordy_y = left_mapper->get_coordy_y ();
    XCAM_FAIL_RETURN (
        ERROR, _left_coordx_y.ptr () && _left_coordy_y.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get left coordinate Y buffers");

    _right_coordx_y = right_mapper->get_coordx_y ();
    _right_coordy_y = right_mapper->get_coordy_y ();
    XCAM_ASSERT (_right_coordx_y.ptr () && _right_coordy_y.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _right_coordx_y.ptr () && _right_coordy_y.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get right coordinate Y buffers");

    _left_coordx_uv = left_mapper->get_coordx_uv ();
    _left_coordy_uv = left_mapper->get_coordy_uv ();
    XCAM_FAIL_RETURN (
        ERROR, _left_coordx_uv.ptr () && _left_coordy_uv.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get left coordinate UV buffers");

    _right_coordx_uv = right_mapper->get_coordx_uv ();
    _right_coordy_uv = right_mapper->get_coordy_uv ();
    XCAM_ASSERT (_right_coordx_uv.ptr () && _right_coordy_uv.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _right_coordx_uv.ptr () && _right_coordy_uv.ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-fastmap_blender failed to get right coordinate UV buffers");

    _mask_y = blender->get_layer0_mask ();

    return XCAM_RETURN_NO_ERROR;
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

XCamReturn
Impl::init_mask_uv ()
{
    XCAM_ASSERT (_mask_y.ptr ());

    const GLBufferDesc &desc_y = _mask_y->get_buffer_desc ();

    GLBufferDesc desc_uv;
    desc_uv.width = desc_y.width / 2;
    desc_uv.height = 1;
    desc_uv.size = desc_uv.width * sizeof (uint8_t);

    SmartPtr<GLBuffer> mask_uv = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, desc_uv.size);
    XCAM_ASSERT (mask_uv.ptr ());
    mask_uv->set_buffer_desc (desc_uv);

    uint8_t *uv_ptr = (uint8_t *) mask_uv->map_range (0, desc_uv.size, GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, uv_ptr, XCAM_RETURN_ERROR_PARAM, "gl-fastmap_blender map mask_uv range failed");

    uint8_t *y_ptr = (uint8_t *) _mask_y->map_range (0, desc_y.size, GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, y_ptr, XCAM_RETURN_ERROR_PARAM, "gl-fastmap_blender map mask_y range failed");

    for (uint32_t i = 0; i < desc_uv.width; ++i) {
        uv_ptr[i] = y_ptr[i * 2];
    }
    _mask_y->unmap ();
    mask_uv->unmap ();

    _mask_uv = mask_uv;

    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<GLImageShader>
create_shader (ShaderID id, const char *prog_name)
{
    SmartPtr<GLImageShader> shader = new GLImageShader (shaders_info[id].name);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shaders_info[id], prog_name);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, NULL,
        "gl-fastmap_blender create compute program for %s failed", shaders_info[id].name);

    return shader;
}

XCamReturn
ImplNV12::init_shaders ()
{
    _shader_y = create_shader (ShaderFastmapBlendY, "fastmap_blend_program_nv12_y");
    _shader_uv = create_shader (ShaderFastmapBlendUVNV12, "fastmap_blend_program_nv12_uv");
    XCAM_ASSERT (_shader_y.ptr () && _shader_uv.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImplNV12::start_y (
    const SmartPtr<GLBuffer> &in0_buf,
    const SmartPtr<GLBuffer> &in1_buf,
    const SmartPtr<GLBuffer> &out_buf)
{
    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 0, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (_left_coordx_y, 1));
    cmds.push_back (new GLCmdBindBufRange (_left_coordy_y, 2));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 3, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (_right_coordx_y, 4));
    cmds.push_back (new GLCmdBindBufRange (_right_coordy_y, 5));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 6, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (_mask_y, 7));
    _shader_y->set_commands (cmds);

    return _shader_y->work (NULL);
}

XCamReturn
ImplNV12::start_uv (
    const SmartPtr<GLBuffer> &in0_buf,
    const SmartPtr<GLBuffer> &in1_buf,
    const SmartPtr<GLBuffer> &out_buf)
{
    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 0, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_left_coordx_uv, 1));
    cmds.push_back (new GLCmdBindBufRange (_left_coordy_uv, 2));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 3, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_right_coordx_uv, 4));
    cmds.push_back (new GLCmdBindBufRange (_right_coordy_uv, 5));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 6, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (_mask_uv, 7));
    _shader_uv->set_commands (cmds);

    return _shader_uv->work (NULL);
}

XCamReturn
ImplNV12::fix_y_parameters (
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
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_info.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_offset_x", blend_area.pos_x / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width0", left_coord_desc.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width1", right_coord_desc.width / unit_bytes));
    _shader_y->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (blend_area.width / unit_bytes, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (blend_area.height, 8) / 8;
    groups_size.z = 1;
    _shader_y->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImplNV12::fix_uv_parameters (
    const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param)
{
    const VideoBufferInfo &left_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &right_info = param->in1_buf->get_video_info ();
    const VideoBufferInfo &out_info = param->out_buf->get_video_info ();
    const GLBufferDesc &left_coord_desc = _left_coordx_uv->get_buffer_desc ();
    const GLBufferDesc &right_coord_desc = _right_coordx_uv->get_buffer_desc ();
    const Rect &blend_area = blender->get_merge_window ();

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width0 = left_info.width / unit_bytes;
    uint32_t in_img_width1 = right_info.width / unit_bytes;
    uint32_t out_img_width = (out_info.width / 2) / unit_bytes;
    uint32_t out_offset_x = (blend_area.pos_x / 2) / unit_bytes;
    uint32_t blend_width = (blend_area.width / 2) / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width0", in_img_width0));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width1", in_img_width1));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_offset_x", out_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width0", left_coord_desc.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width1", right_coord_desc.width / unit_bytes));
    _shader_uv->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (blend_width, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (blend_area.height / 2, 4) / 4;
    groups_size.z = 1;
    _shader_uv->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImplYUV420::init_shaders ()
{
    _shader_y = create_shader (ShaderFastmapBlendY, "fastmap_blend_program_yuv420_y");
    _shader_uv = create_shader (ShaderFastmapBlendUVYUV420, "fastmap_blend_program_yuv420_uv");
    XCAM_ASSERT (_shader_y.ptr () && _shader_uv.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImplYUV420::start_y (
    const SmartPtr<GLBuffer> &in0_buf,
    const SmartPtr<GLBuffer> &in1_buf,
    const SmartPtr<GLBuffer> &out_buf)
{
    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 0, YUV420PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (_left_coordx_y, 1));
    cmds.push_back (new GLCmdBindBufRange (_left_coordy_y, 2));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 3, YUV420PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (_right_coordx_y, 4));
    cmds.push_back (new GLCmdBindBufRange (_right_coordy_y, 5));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 6, YUV420PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (_mask_y, 7));
    _shader_y->set_commands (cmds);

    return _shader_y->work (NULL);
};

XCamReturn
ImplYUV420::start_uv(
    const SmartPtr<GLBuffer> &in0_buf,
    const SmartPtr<GLBuffer> &in1_buf,
    const SmartPtr<GLBuffer> &out_buf)
{
    GLCmdList cmds;
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 0, YUV420PlaneUIdx));
    cmds.push_back (new GLCmdBindBufRange (in0_buf, 1, YUV420PlaneVIdx));
    cmds.push_back (new GLCmdBindBufRange (_left_coordx_uv, 2));
    cmds.push_back (new GLCmdBindBufRange (_left_coordy_uv, 3));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 4, YUV420PlaneUIdx));
    cmds.push_back (new GLCmdBindBufRange (in1_buf, 5, YUV420PlaneVIdx));
    cmds.push_back (new GLCmdBindBufRange (_right_coordx_uv, 6));
    cmds.push_back (new GLCmdBindBufRange (_right_coordy_uv, 7));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 8, YUV420PlaneUIdx));
    cmds.push_back (new GLCmdBindBufRange (out_buf, 9, YUV420PlaneVIdx));
    cmds.push_back (new GLCmdBindBufRange (_mask_uv, 10));
    _shader_uv->set_commands (cmds);

    return _shader_uv->work (NULL);
};

XCamReturn
ImplYUV420::fix_y_parameters (
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
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_info.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_offset_x", blend_area.pos_x / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width0", left_coord_desc.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width1", right_coord_desc.width / unit_bytes));
    _shader_y->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (blend_area.width / unit_bytes, 4) / 4;
    groups_size.y = XCAM_ALIGN_UP (blend_area.height, 8) / 8;
    groups_size.z = 1;
    _shader_y->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImplYUV420::fix_uv_parameters (
    const SmartPtr<GLBlender> &blender, const SmartPtr<GLBlender::BlenderParam> &param)
{
    const VideoBufferInfo &left_info = param->in_buf->get_video_info ();
    const VideoBufferInfo &right_info = param->in1_buf->get_video_info ();
    const VideoBufferInfo &out_info = param->out_buf->get_video_info ();
    const GLBufferDesc &left_coord_desc = _left_coordx_uv->get_buffer_desc ();
    const GLBufferDesc &right_coord_desc = _right_coordx_uv->get_buffer_desc ();
    const Rect &blend_area = blender->get_merge_window ();

    const size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width0 = (left_info.width / 2) / unit_bytes;
    uint32_t in_img_width1 = (right_info.width / 2) / unit_bytes;
    uint32_t out_img_width = (out_info.width / 2) / unit_bytes;
    uint32_t out_offset_x = (blend_area.pos_x / 2) / unit_bytes;
    uint32_t blend_width = (blend_area.width / 2) / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width0", in_img_width0));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width1", in_img_width1));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_offset_x", out_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width0", left_coord_desc.width / unit_bytes));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("coords_width1", right_coord_desc.width / unit_bytes));
    _shader_uv->set_commands (cmds);

    GLGroupsSize groups_size_uv;
    groups_size_uv.x = XCAM_ALIGN_UP (blend_width, 4) / 4;
    groups_size_uv.y = XCAM_ALIGN_UP (blend_area.height / 2, 8) / 8;
    groups_size_uv.z = 1;
    _shader_uv->set_groups_size (groups_size_uv);

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

    const VideoBufferInfo &info = blend_param->in_buf->get_video_info ();
    SmartPtr<GLFastmapBlendPriv::Impl> impl;
    if (info.format == V4L2_PIX_FMT_NV12) {
        impl = new GLFastmapBlendPriv::ImplNV12 ();
    } else {
        impl = new GLFastmapBlendPriv::ImplYUV420 ();
    }

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
