/*
 * gl_blender.cpp - gl blender implementation
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

#include "xcam_utils.h"
#include "gl_image_shader.h"
#include "gl_utils.h"
#include "gl_blender.h"
#include "gl_sync.h"

#define DUMP_BUFFER 0

#define GL_BLENDER_ALIGN_X 8
#define GL_BLENDER_ALIGN_Y 4

#define PYR_BOTTOM_LAYER 1

#define GL_RECON_POOL_SIZE 4
#define GL_GS_POOL_SIZE (2*GL_RECON_POOL_SIZE)
#define GL_LAP_POOL_SIZE GL_GS_POOL_SIZE

#define GAUSS_RADIUS 2
#define GAUSS_DIAMETER  ((GAUSS_RADIUS)*2+1)

const float gauss_coeffs[GAUSS_DIAMETER] = {0.152f, 0.222f, 0.252f, 0.222f, 0.152f};

namespace XCam {

enum BufIdx {
    BufIdx0 = 0,
    BufIdx1,
    BufIdxMax
};

#if DUMP_BUFFER
static void
dump_level_buf (
    const SmartPtr<GLBuffer> &buf, const char *name, uint32_t level, BufIdx idx = BufIdxMax)
{
    char file_name[256];
    if (idx == BufIdxMax)
        snprintf (file_name, 256, "%s-L%d", name, level);
    else
        snprintf (file_name, 256, "%s-L%d-Idx%d", name, level, idx);
    dump_buf (buf, file_name);
}
#endif

namespace GLBlenderPriv {

enum ShaderID {
    ShaderGaussScalePyr = 0,
    ShaderLapTransPyr,
    ShaderBlendPyr,
    ShaderReconstructPyr,
    ShaderYUV420GaussScalePyr,
    ShaderYUV420LapTransPyr,
    ShaderYUV420BlendPyr,
    ShaderYUV420ReconstructPyr
};

static const GLShaderInfo shaders_info[] = {
    {
        GL_COMPUTE_SHADER,
        "shader_gauss_scale_pyr",
#include "shader_gauss_scale_pyr.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_lap_trans_pyr",
#include "shader_lap_trans_pyr.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_blend_pyr",
#include "shader_blend_pyr.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_reconstruct_pyr",
#include "shader_reconstruct_pyr.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_gauss_scale_pyr_yuv420",
#include "shader_gauss_scale_pyr_yuv420.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_lap_trans_pyr_yuv420",
#include "shader_lap_trans_pyr_yuv420.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_blend_pyr_yuv420",
#include "shader_blend_pyr_yuv420.comp.slx"
        , 0
    },
    {
        GL_COMPUTE_SHADER,
        "shader_reconstruct_pyr_yuv420",
#include "shader_reconstruct_pyr_yuv420.comp.slx"
        , 0
    }
};

struct PyrLayer {
    uint32_t                   blend_width;
    uint32_t                   blend_height;

    SmartPtr<GLImageShader>    gauss_scale[BufIdxMax];
    SmartPtr<GLImageShader>    lap_trans[BufIdxMax];
    SmartPtr<GLImageShader>    blend;
    SmartPtr<GLImageShader>    reconstruct;

    SmartPtr<BufferPool>       gs_pool;
    SmartPtr<BufferPool>       lap_pool;
    SmartPtr<BufferPool>       reconstruct_pool;

    SmartPtr<GLBuffer>         gs_buf[BufIdxMax];
    SmartPtr<GLBuffer>         lap_buf[BufIdxMax];
    SmartPtr<GLBuffer>         blend_buf;
    SmartPtr<GLBuffer>         reconstruct_buf;

    SmartPtr<GLBuffer>         mask;

    PyrLayer () : blend_width (0), blend_height (0) {}
};

class BlenderImpl {
public:
    PyrLayer                      _pyr_layer[XCAM_GL_PYRAMID_MAX_LEVEL];
    uint32_t                      _pyr_layers_num;

private:
    GLBlender                    *_blender;

    Rect                          _in_area[BufIdxMax];
    Rect                          _out_area;

    uint32_t                      _in_width[BufIdxMax];
    uint32_t                      _in_height[BufIdxMax];
    uint32_t                      _out_width;
    uint32_t                      _out_height;

    uint32_t                      _pix_fmt;
    bool                          _is_nv12_fmt;

public:
    BlenderImpl (GLBlender *blender, uint32_t level);

    XCamReturn init_parameters (
        const VideoBufferInfo &in0_info, const VideoBufferInfo &in1_info);
    XCamReturn init_buffers ();
    XCamReturn create_shaders ();
    XCamReturn fix_parameters ();

    XCamReturn update_buffers (
        const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, const SmartPtr<VideoBuffer> &out);

    XCamReturn start_gauss_scale (uint32_t level, BufIdx idx);
    XCamReturn start_lap_trans (uint32_t level, BufIdx idx);
    XCamReturn start_blend ();
    XCamReturn start_reconstruct (uint32_t level);

    XCamReturn stop ();

private:
    XCamReturn fix_gs_params (uint32_t level, BufIdx idx);
    XCamReturn fix_lap_params (uint32_t level, BufIdx idx);
    XCamReturn fix_blend_params ();
    XCamReturn fix_reconstruct_params (uint32_t level);

    XCamReturn bind_io_bufs_to_layer0 (
        const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, const SmartPtr<VideoBuffer> &out);

    XCamReturn init_layer0_mask ();
    XCamReturn scale_down_mask (uint32_t level);

};

BlenderImpl::BlenderImpl (GLBlender *blender, uint32_t level)
    : _pyr_layers_num (level)
    , _blender (blender)
    , _out_width (0)
    , _out_height (0)
    , _pix_fmt (V4L2_PIX_FMT_NV12)
    , _is_nv12_fmt (true)
{
    xcam_mem_clear (_in_width);
    xcam_mem_clear (_in_height);
}

SmartPtr<GLImageShader>
create_pyr_shader (ShaderID id)
{
    const GLShaderInfo &info = shaders_info[id];

    SmartPtr<GLImageShader> shader = new GLImageShader (info.name);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (info);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, NULL,
        "gl-blender create %s program failed", info.name);

    return shader;
}

static XCamReturn
check_blend_area (const SmartPtr<GLBlender> &blender)
{
    const Rect &in0_area = blender->get_input_merge_area (BufIdx0);
    XCAM_FAIL_RETURN (
        ERROR,
        in0_area.pos_y == 0 && in0_area.width && in0_area.height &&
        in0_area.pos_x % GL_BLENDER_ALIGN_X == 0 &&
        in0_area.width % GL_BLENDER_ALIGN_X == 0 &&
        in0_area.height % GL_BLENDER_ALIGN_Y == 0,
        XCAM_RETURN_ERROR_PARAM,
        "gl-blender invalid input0 merge area, pos_x: %d, pos_y: %d, width: %d, height: %d",
        in0_area.pos_x, in0_area.pos_y, in0_area.width, in0_area.height);

    const Rect &in1_area = blender->get_input_merge_area (BufIdx1);
    XCAM_FAIL_RETURN (
        ERROR,
        in1_area.pos_y == 0 && in1_area.width && in1_area.height &&
        in1_area.pos_x % GL_BLENDER_ALIGN_X == 0 &&
        in1_area.width % GL_BLENDER_ALIGN_X == 0 &&
        in1_area.height % GL_BLENDER_ALIGN_Y == 0,
        XCAM_RETURN_ERROR_PARAM,
        "gl-blender invalid input1 merge area, pos_x: %d, pos_y: %d, width: %d, height: %d",
        in1_area.pos_x, in1_area.pos_y, in1_area.width, in1_area.height);

    const Rect &out_area = blender->get_merge_window ();
    XCAM_FAIL_RETURN (
        ERROR,
        out_area.pos_y == 0 && out_area.width && out_area.height &&
        out_area.pos_x % GL_BLENDER_ALIGN_X == 0 &&
        out_area.width % GL_BLENDER_ALIGN_X == 0 &&
        out_area.height % GL_BLENDER_ALIGN_Y == 0,
        XCAM_RETURN_ERROR_PARAM,
        "gl-blender invalid output merge area, pos_x: %d, pos_y: %d, width: %d, height: %d",
        out_area.pos_x, out_area.pos_y, out_area.width, out_area.height);

    XCAM_FAIL_RETURN (
        ERROR,
        in0_area.width == in1_area.width && in0_area.height == in1_area.height &&
        in0_area.width == out_area.width && in0_area.height == out_area.height,
        XCAM_RETURN_ERROR_PARAM,
        "gl-blender invalid input or output overlap area, input0: %dx%d, input1: dx%d, output: %dx%d",
        in0_area.width, in0_area.height, in1_area.width, in1_area.height, out_area.width, out_area.height);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::init_parameters (
    const VideoBufferInfo &in0_info, const VideoBufferInfo &in1_info)
{
    XCamReturn ret = check_blend_area (_blender);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender check blend area failed");

    _in_area[BufIdx0] = _blender->get_input_merge_area (BufIdx0);
    _in_area[BufIdx1] = _blender->get_input_merge_area (BufIdx1);
    _out_area = _blender->get_merge_window ();

    _in_width[BufIdx0] = in0_info.width;
    _in_height[BufIdx0] = in0_info.height;
    _in_width[BufIdx1] = in1_info.width;
    _in_height[BufIdx1] = in1_info.height;
    _blender->get_output_size (_out_width, _out_height);

    _pyr_layer[0].blend_width = _out_area.width;
    _pyr_layer[0].blend_height = _out_area.height;

    for (uint32_t i = PYR_BOTTOM_LAYER; i < _pyr_layers_num; ++i) {
        PyrLayer &prev_layer = _pyr_layer[i - 1];
        PyrLayer &layer = _pyr_layer[i];

        layer.blend_width = XCAM_ALIGN_UP ((prev_layer.blend_width + 1) / 2, GL_BLENDER_ALIGN_X);
        layer.blend_height = XCAM_ALIGN_UP ((prev_layer.blend_height + 1) / 2, GL_BLENDER_ALIGN_Y);
    }

    _pix_fmt = in0_info.format;
    _is_nv12_fmt = (_pix_fmt == V4L2_PIX_FMT_NV12);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::init_buffers ()
{
    XCamReturn ret = init_layer0_mask ();
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender init layer0 mask failed");

    VideoBufferInfo info;
    SmartPtr<BufferPool> pool;
    for (uint32_t i = PYR_BOTTOM_LAYER; i < _pyr_layers_num; ++i) {
        ret = scale_down_mask (i);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-blender scale down mask failed, level: %d", i);

        PyrLayer &layer = _pyr_layer[i];
        info.init (_pix_fmt, layer.blend_width, layer.blend_height);

        pool = new GLVideoBufferPool (info);
        XCAM_FAIL_RETURN (
            ERROR, pool.ptr () && pool->reserve (GL_GS_POOL_SIZE), XCAM_RETURN_ERROR_MEM,
            "gl-blender reserve gauss scale buffer pool failed, buffer size: %dx%d",
            layer.blend_width, layer.blend_height);
        layer.gs_pool = pool;

        pool = new GLVideoBufferPool (info);
        XCAM_FAIL_RETURN (
            ERROR, pool.ptr () && pool->reserve (GL_RECON_POOL_SIZE), XCAM_RETURN_ERROR_MEM,
            "gl-blender reserve reconstruct buffer pool failed, buffer size: %dx%d",
            layer.blend_width, layer.blend_height);
        layer.reconstruct_pool = pool;

        PyrLayer &prev_layer = _pyr_layer[i - 1];
        info.init (_pix_fmt, prev_layer.blend_width, prev_layer.blend_height);

        pool = new GLVideoBufferPool (info);
        XCAM_FAIL_RETURN (
            ERROR, pool.ptr () && pool->reserve (GL_LAP_POOL_SIZE), XCAM_RETURN_ERROR_MEM,
            "gl-blender reserve laplace transformation buffer pool failed, buffer size: %dx%d",
            prev_layer.blend_width, prev_layer.blend_height);
        layer.lap_pool = pool;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::create_shaders ()
{
    PyrLayer &top_layer = _pyr_layer[_pyr_layers_num - 1];
    top_layer.blend = _is_nv12_fmt ?
        create_pyr_shader (ShaderBlendPyr) : create_pyr_shader (ShaderYUV420BlendPyr);
    XCAM_ASSERT (top_layer.blend.ptr ());

    for (uint32_t i = PYR_BOTTOM_LAYER; i < _pyr_layers_num; ++i) {
        PyrLayer &layer = _pyr_layer[i];
        if (_is_nv12_fmt) {
            layer.gauss_scale[BufIdx0] = create_pyr_shader (ShaderGaussScalePyr);
            layer.gauss_scale[BufIdx1] = create_pyr_shader (ShaderGaussScalePyr);
            layer.lap_trans[BufIdx0] = create_pyr_shader (ShaderLapTransPyr);
            layer.lap_trans[BufIdx1] = create_pyr_shader (ShaderLapTransPyr);
            layer.reconstruct = create_pyr_shader (ShaderReconstructPyr);
        } else {
            layer.gauss_scale[BufIdx0] = create_pyr_shader (ShaderYUV420GaussScalePyr);
            layer.gauss_scale[BufIdx1] = create_pyr_shader (ShaderYUV420GaussScalePyr);
            layer.lap_trans[BufIdx0] = create_pyr_shader (ShaderYUV420LapTransPyr);
            layer.lap_trans[BufIdx1] = create_pyr_shader (ShaderYUV420LapTransPyr);
            layer.reconstruct = create_pyr_shader (ShaderYUV420ReconstructPyr);
        }
        XCAM_ASSERT (layer.gauss_scale[BufIdx0].ptr () && layer.gauss_scale[BufIdx1].ptr ());
        XCAM_ASSERT (layer.lap_trans[BufIdx0].ptr () && layer.lap_trans[BufIdx1].ptr ());
        XCAM_ASSERT (layer.reconstruct.ptr ());
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_parameters ()
{
    for (uint32_t level = PYR_BOTTOM_LAYER; level < _pyr_layers_num; ++level) {
        fix_gs_params (level, BufIdx0);
        fix_gs_params (level, BufIdx1);
        fix_lap_params (level, BufIdx0);
        fix_lap_params (level, BufIdx1);
        fix_reconstruct_params (level);
    }

    fix_blend_params ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::update_buffers (
    const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, const SmartPtr<VideoBuffer> &out)
{
    bind_io_bufs_to_layer0 (in0, in1, out);

    SmartPtr<VideoBuffer> buf;
    for (uint32_t level = PYR_BOTTOM_LAYER; level < _pyr_layers_num; ++level) {
        PyrLayer &layer = _pyr_layer[level];

        buf = layer.gs_pool->get_buffer ();
        layer.gs_buf[BufIdx0] = get_glbuffer (buf);
        buf = layer.gs_pool->get_buffer ();
        layer.gs_buf[BufIdx1] = get_glbuffer (buf);

        buf = layer.lap_pool->get_buffer ();
        layer.lap_buf[BufIdx0] = get_glbuffer (buf);
        buf = layer.lap_pool->get_buffer ();
        layer.lap_buf[BufIdx1] = get_glbuffer (buf);

        buf = layer.reconstruct_pool->get_buffer ();
        layer.reconstruct_buf = get_glbuffer (buf);

        layer.blend_buf = layer.reconstruct_buf;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::start_gauss_scale (uint32_t level, BufIdx idx)
{
    PyrLayer &prev_layer = _pyr_layer[level - 1];
    PyrLayer &layer = _pyr_layer[level];

    GLCmdList cmds;
    if (_is_nv12_fmt) {
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 0, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 1, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 2, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 3, NV12PlaneUVIdx));
    } else {
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 0, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 1, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 2, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 3, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 4, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 5, YUV420PlaneVIdx));
    }
    layer.gauss_scale[idx]->set_commands (cmds);

    return layer.gauss_scale[idx]->work (NULL);
}

XCamReturn
BlenderImpl::start_lap_trans (uint32_t level, BufIdx idx)
{
    PyrLayer &prev_layer = _pyr_layer[level - 1];
    PyrLayer &layer = _pyr_layer[level];

    GLCmdList cmds;
    if (_is_nv12_fmt) {
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 0, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 1, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 2, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 3, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[idx], 4, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[idx], 5, NV12PlaneUVIdx));
    } else {
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 0, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 1, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.gs_buf[idx], 2, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 3, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 4, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[idx], 5, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[idx], 6, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[idx], 7, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[idx], 8, YUV420PlaneVIdx));
    }
    layer.lap_trans[idx]->set_commands (cmds);

    return layer.lap_trans[idx]->work (NULL);
}

XCamReturn
BlenderImpl::start_blend ()
{
    PyrLayer &layer = _pyr_layer[_pyr_layers_num - 1];

    GLCmdList cmds;
    if (_is_nv12_fmt) {
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx0], 0, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx0], 1, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx1], 2, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx1], 3, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 4, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 5, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufBase (layer.mask, 6));
    } else {
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx0], 0, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx0], 1, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx0], 2, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx1], 3, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx1], 4, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.gs_buf[BufIdx1], 5, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 6, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 7, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 8, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufBase (layer.mask, 9));
    }
    layer.blend->set_commands (cmds);

    return layer.blend->work (NULL);
}

XCamReturn
BlenderImpl::start_reconstruct (uint32_t level)
{
    PyrLayer &prev_layer = _pyr_layer[level - 1];
    PyrLayer &layer = _pyr_layer[level];

    GLCmdList cmds;
    if (_is_nv12_fmt) {
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx0], 0, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx0], 1, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx1], 2, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx1], 3, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.blend_buf, 4, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.blend_buf, 5, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 6, NV12PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 7, NV12PlaneUVIdx));
        cmds.push_back (new GLCmdBindBufBase (prev_layer.mask, 8));
    } else {
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx0], 0, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx0], 1, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx0], 2, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx1], 3, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx1], 4, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.lap_buf[BufIdx1], 5, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.blend_buf, 6, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.blend_buf, 7, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (prev_layer.blend_buf, 8, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 9, YUV420PlaneYIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 10, YUV420PlaneUIdx));
        cmds.push_back (new GLCmdBindBufRange (layer.blend_buf, 11, YUV420PlaneVIdx));
        cmds.push_back (new GLCmdBindBufBase (prev_layer.mask, 12));
    }
    layer.reconstruct->set_commands (cmds);

    return layer.reconstruct->work (NULL);
}

XCamReturn
BlenderImpl::stop ()
{
    for (uint32_t i = 0; i < _pyr_layers_num; ++i) {
        PyrLayer &layer = _pyr_layer[i];
        layer.gauss_scale[BufIdx0].release ();
        layer.gauss_scale[BufIdx1].release ();
        layer.lap_trans[BufIdx0].release ();
        layer.lap_trans[BufIdx1].release ();
        layer.blend.release ();
        layer.reconstruct.release ();

        layer.gs_pool.release ();
        layer.lap_pool.release ();
        layer.reconstruct_pool.release ();

        layer.mask.release ();
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_gs_params (uint32_t level, BufIdx idx)
{
    XCAM_ASSERT (level >= PYR_BOTTOM_LAYER && level < _pyr_layers_num);

    PyrLayer &prev_layer = _pyr_layer[level - 1];
    PyrLayer &layer = _pyr_layer[level];
    SmartPtr<GLImageShader> &gauss_scale = layer.gauss_scale[idx];

    const size_t unit_bytes = sizeof (uint32_t) * (_is_nv12_fmt ? 1 : 4);
    bool bottom_layer = level == PYR_BOTTOM_LAYER;
    uint32_t in_img_width = (bottom_layer ? _in_width[idx] : prev_layer.blend_width) / unit_bytes;
    uint32_t in_offset_x = (bottom_layer ? _in_area[idx].pos_x : 0) / unit_bytes;
    uint32_t out_img_width = layer.blend_width / unit_bytes * (_is_nv12_fmt ? 1 : 2);
    uint32_t merge_width = prev_layer.blend_width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", prev_layer.blend_height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_offset_x", in_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("merge_width", merge_width));
    gauss_scale->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (out_img_width, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (layer.blend_height, 16) / 16;
    groups_size.z = 1;
    gauss_scale->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_lap_params (uint32_t level, BufIdx idx)
{
    XCAM_ASSERT (level >= PYR_BOTTOM_LAYER && level < _pyr_layers_num);

    PyrLayer &prev_layer = _pyr_layer[level - 1];
    PyrLayer &layer = _pyr_layer[level];
    SmartPtr<GLImageShader> &lap_trans = layer.lap_trans[idx];

    const size_t unit_bytes = sizeof (uint32_t) * (_is_nv12_fmt ? 2 : 4);
    bool bottom_layer = level == PYR_BOTTOM_LAYER;
    uint32_t in_img_width = (bottom_layer? _in_width[idx] : prev_layer.blend_width) / unit_bytes;
    uint32_t in_offset_x = (bottom_layer ? _in_area[idx].pos_x : 0) / unit_bytes;
    uint32_t gaussscale_img_width = layer.blend_width / unit_bytes * 2;
    uint32_t merge_width = prev_layer.blend_width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", prev_layer.blend_height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_offset_x", in_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("gaussscale_img_width", gaussscale_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("gaussscale_img_height", layer.blend_height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("merge_width", merge_width));
    lap_trans->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (merge_width, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (prev_layer.blend_height, 32) / 32;
    groups_size.z = 1;
    lap_trans->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_blend_params ()
{
    PyrLayer &layer = _pyr_layer[_pyr_layers_num - 1];

    const size_t unit_bytes = sizeof (uint32_t) * (_is_nv12_fmt ? 2 : 4);
    bool single_layer = _pyr_layers_num == PYR_BOTTOM_LAYER;
    uint32_t in0_img_width = (single_layer ? _in_width[BufIdx0] : layer.blend_width) / unit_bytes;
    uint32_t in1_img_width = (single_layer ? _in_width[BufIdx1] : layer.blend_width) / unit_bytes;
    uint32_t out_img_width = (single_layer ? _out_width : layer.blend_width) / unit_bytes;
    uint32_t in0_offset_x = (single_layer ? _in_area[BufIdx0].pos_x : 0) / unit_bytes;
    uint32_t in1_offset_x = (single_layer ? _in_area[BufIdx1].pos_x : 0) / unit_bytes;
    uint32_t out_offset_x = (single_layer ? _out_area.pos_x : 0) / unit_bytes;
    uint32_t blend_width = layer.blend_width / unit_bytes;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in0_img_width", in0_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in1_img_width", in1_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in0_offset_x", in0_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in1_offset_x", in1_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_offset_x", out_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("blend_width", blend_width));
    layer.blend->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (blend_width, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (layer.blend_height, 16) / 16;
    groups_size.z = 1;
    layer.blend->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::fix_reconstruct_params (uint32_t level)
{
    XCAM_ASSERT (level >= PYR_BOTTOM_LAYER && level < _pyr_layers_num);

    PyrLayer &prev_layer = _pyr_layer[level - 1];
    PyrLayer &layer = _pyr_layer[level];
    SmartPtr<GLImageShader> &reconstruct = layer.reconstruct;

    const size_t unit_bytes = sizeof (uint32_t) * (_is_nv12_fmt ? 2 : 4);
    bool bottom_layer = level == PYR_BOTTOM_LAYER;
    uint32_t lap_img_width = prev_layer.blend_width / unit_bytes;
    uint32_t out_img_width = (bottom_layer ? _out_width : prev_layer.blend_width) / unit_bytes;;
    uint32_t out_offset_x = (bottom_layer ? _out_area.pos_x : 0) / unit_bytes;
    uint32_t prev_blend_img_width = layer.blend_width / unit_bytes * 2;

    GLCmdList cmds;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lap_img_width", lap_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lap_img_height", prev_layer.blend_height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_offset_x", out_offset_x));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("prev_blend_img_width", prev_blend_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("prev_blend_img_height", layer.blend_height));
    reconstruct->set_commands (cmds);

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (lap_img_width, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (prev_layer.blend_height, 32) / 32;
    groups_size.z = 1;
    reconstruct->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::bind_io_bufs_to_layer0 (
    const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, const SmartPtr<VideoBuffer> &out)
{
    PyrLayer &layer0 = _pyr_layer[0];
    layer0.gs_buf[BufIdx0] = get_glbuffer (in0);
    layer0.gs_buf[BufIdx1] = get_glbuffer (in1);

    if (out.ptr ()) {
        layer0.reconstruct_buf = get_glbuffer (out);
        layer0.blend_buf = layer0.reconstruct_buf;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::init_layer0_mask ()
{
    PyrLayer &layer = _pyr_layer[0];
    XCAM_ASSERT (layer.blend_width && (layer.blend_width % GL_BLENDER_ALIGN_X == 0));

    uint32_t buf_size = layer.blend_width * sizeof (uint8_t);
    SmartPtr<GLBuffer> buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, buf_size);
    XCAM_ASSERT (buf.ptr ());

    GLBufferDesc desc;
    desc.width = layer.blend_width;
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

    layer.mask = buf;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
BlenderImpl::scale_down_mask (uint32_t level)
{
    XCAM_ASSERT (level >= PYR_BOTTOM_LAYER);

    PyrLayer &layer = _pyr_layer[level];
    XCAM_ASSERT (layer.blend_width && (layer.blend_width % GL_BLENDER_ALIGN_X == 0));

    uint32_t buf_size = layer.blend_width * sizeof (uint8_t);
    SmartPtr<GLBuffer> buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, buf_size);
    XCAM_ASSERT (buf.ptr ());

    GLBufferDesc desc;
    desc.width = layer.blend_width;
    desc.height = 1;
    desc.size = buf_size;
    buf->set_buffer_desc (desc);

    SmartPtr<GLBuffer> &prev_mask = _pyr_layer[level - 1].mask;
    XCAM_ASSERT (prev_mask.ptr ());

    const GLBufferDesc prev_desc = prev_mask->get_buffer_desc ();
    uint8_t *prev_ptr = (uint8_t *) prev_mask->map_range (0, prev_desc.size, GL_MAP_READ_BIT);
    XCAM_FAIL_RETURN (ERROR, prev_ptr, XCAM_RETURN_ERROR_PARAM, "gl-blender map range failed");

    uint8_t *cur_ptr = (uint8_t *) buf->map_range (0, desc.size, GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, cur_ptr, XCAM_RETURN_ERROR_PARAM, "gl-blender map range failed");

    for (uint32_t i = 0; i < desc.width; ++i) {
        int prev_start = i * 2 - 2;
        float sum = 0.0f;

        for (int j = 0; j < GAUSS_DIAMETER; ++j) {
            int prev_idx = XCAM_CLAMP (prev_start + j, 0, (int)prev_desc.width);
            sum += prev_ptr[prev_idx] * gauss_coeffs[j];
        }

        cur_ptr[i] = XCAM_CLAMP (sum, 0.0f, 255.0f);
    }

    buf->unmap ();
    prev_mask->unmap ();

    layer.mask = buf;

    return XCAM_RETURN_NO_ERROR;
}

}

GLBlender::GLBlender (const char *name)
    : GLImageHandler (name)
    , Blender (GL_BLENDER_ALIGN_X, GL_BLENDER_ALIGN_Y)
{
    SmartPtr<GLBlenderPriv::BlenderImpl> impl =
        new GLBlenderPriv::BlenderImpl (this, 1);
    XCAM_ASSERT (impl.ptr ());

    _impl = impl;
}

GLBlender::~GLBlender ()
{
}

bool
GLBlender::set_pyr_levels (uint32_t levels)
{
    XCAM_FAIL_RETURN (
        ERROR, levels > 0 && levels <= XCAM_GL_PYRAMID_MAX_LEVEL, false,
        "gl-blender invalid levels number: %d, levels number must be in (0, %d]",
        levels, XCAM_GL_PYRAMID_MAX_LEVEL);

    _impl->_pyr_layers_num = levels;

    return true;
}

const SmartPtr<GLBuffer> &
GLBlender::get_layer0_mask () const
{
    return _impl->_pyr_layer[0].mask;
}

XCamReturn
GLBlender::terminate ()
{
    _impl->stop ();
    return GLImageHandler::terminate ();
}

XCamReturn
GLBlender::blend (
    const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, SmartPtr<VideoBuffer> &out)
{
    XCAM_ASSERT (in0.ptr () && in1.ptr ());

    SmartPtr<BlenderParam> param = new BlenderParam (in0, in1, out);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, true);

    GLSync::flush ();
    if (xcam_ret_is_ok (ret) && !out.ptr ()) {
        out = param->out_buf;
    }

    return ret;
}

XCamReturn
GLBlender::start_work (const SmartPtr<ImageHandler::Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());

    SmartPtr<BlenderParam> param = base.dynamic_cast_ptr<BlenderParam> ();
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->in1_buf.ptr () && param->out_buf.ptr ());

    _impl->update_buffers (param->in_buf, param->in1_buf, param->out_buf);

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    for (uint32_t level = PYR_BOTTOM_LAYER; level < _impl->_pyr_layers_num; ++level) {
        ret = _impl->start_gauss_scale (level, BufIdx0);
        XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender execute gauss scale failed");
        ret = _impl->start_gauss_scale (level, BufIdx1);
        XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender execute gauss scale failed");

        ret = _impl->start_lap_trans (level, BufIdx0);
        XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender execute lap trans failed");
        ret = _impl->start_lap_trans (level, BufIdx1);
        XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender execute lap trans failed");
    }

    ret = _impl->start_blend ();
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender execute blend failed");

    for (uint32_t level = _impl->_pyr_layers_num - 1; level >= PYR_BOTTOM_LAYER; --level) {
        ret = _impl->start_reconstruct (level);
        XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender execute reconstruct failed");
    }

#if DUMP_BUFFER
    GLSync::finish ();

    dump_buf (_impl->_pyr_layer[0].gs_buf[BufIdx0], "input0");
    dump_buf (_impl->_pyr_layer[0].gs_buf[BufIdx1], "input1");
    dump_buf (_impl->_pyr_layer[0].blend_buf, "output");
    dump_buf (_impl->_pyr_layer[_impl->_pyr_layers_num - 1].blend_buf, "blend");
    for (uint32_t level = PYR_BOTTOM_LAYER; level < _impl->_pyr_layers_num; ++level) {
        GLBlenderPriv::PyrLayer &layer = _impl->_pyr_layer[level];
        dump_level_buf (layer.gs_buf[BufIdx0], "gauss-scale", level, BufIdx0);
        dump_level_buf (layer.gs_buf[BufIdx1], "gauss-scale", level, BufIdx1);
        dump_level_buf (layer.lap_buf[BufIdx0], "lap-trans", level, BufIdx0);
        dump_level_buf (layer.lap_buf[BufIdx1], "lap-trans", level, BufIdx1);
        dump_level_buf (layer.reconstruct_buf, "reconstruct", level);
    }
#endif

    return XCAM_RETURN_NO_ERROR;
};

static XCamReturn
set_output_info (GLBlender *blender, uint32_t format)
{
    uint32_t width, height;
    blender->get_output_size (width, height);
    XCAM_FAIL_RETURN (
        ERROR, width && height, XCAM_RETURN_ERROR_PARAM,
        "gl-blender invalid output size: %dx%d", width, height);

    VideoBufferInfo info;
    info.init (
        format, width, height,
        XCAM_ALIGN_UP (width, GL_BLENDER_ALIGN_X),
        XCAM_ALIGN_UP (height, GL_BLENDER_ALIGN_Y));
    blender->set_out_video_info (info);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLBlender::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (_impl->_pyr_layers_num <= XCAM_GL_PYRAMID_MAX_LEVEL);

    SmartPtr<BlenderParam> blend_param = param.dynamic_cast_ptr<BlenderParam> ();
    XCAM_ASSERT (blend_param.ptr () && blend_param->in_buf.ptr () && blend_param->in1_buf.ptr ());

    const VideoBufferInfo &in0_info = blend_param->in_buf->get_video_info ();
    const VideoBufferInfo &in1_info = blend_param->in1_buf->get_video_info ();

    XCamReturn ret = set_output_info (this, in0_info.format);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender set output info failed");

    ret = _impl->init_parameters (in0_info, in1_info);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender init parameters failed");

    ret = _impl->init_buffers ();
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender init buffers failed");

    ret = _impl->create_shaders ();
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender create shaders failed");

    _impl->fix_parameters ();
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "gl-blender fix parameters failed");

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<GLImageHandler>
create_gl_blender ()
{
    SmartPtr<GLBlender> blender = new GLBlender();
    XCAM_ASSERT (blender.ptr ());
    return blender;
}

SmartPtr<Blender>
Blender::create_gl_blender ()
{
    SmartPtr<GLImageHandler> handler = XCam::create_gl_blender ();
    return handler.dynamic_cast_ptr<Blender> ();
}

}
