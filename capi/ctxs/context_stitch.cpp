/*
 * context_stitch.cpp - private context for image stitching
 *
 *  Copyright (c) 2019 Intel Corporation
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

#include "context_stitch.h"
#include "stitch_params.h"
#include "soft/soft_video_buf_allocator.h"
#if HAVE_GLES
#include "gles/gl_video_buffer.h"
#include "gles/egl/egl_base.h"
#endif
#if HAVE_VULKAN
#include "vulkan/vk_device.h"
#endif

namespace XCam {

typedef struct Pair {
    uint32_t id;
    const char *name;
} Pair;

static const Pair cammodel_pairs[] = {
    {CamA2C1080P, "cama2c1080p"},
    {CamB4C1080P, "camb4c1080p"},
    {CamC3C8K, "camc3c8k"},
    {CamD3C8K, "camd3c8k"},
    {0, NULL}
};

static const Pair module_pairs[] = {
    {StitchNone, "none"},
    {StitchSoft, "soft"},
    {StitchGLES, "gles"},
    {StitchVulkan, "vulkan"},
    {0, NULL}
};

static const Pair dewarp_pairs[] = {
    {DewarpSphere, "sphere"},
    {DewarpBowl, "bowl"},
    {0, NULL}
};

static const Pair scopic_pairs[] = {
    {ScopicMono, "mono"},
    {ScopicStereoLeft, "stereoleft"},
    {ScopicStereoRight, "stereoright"},
    {0, NULL}
};

static const Pair scale_pairs[] = {
    {ScaleSingleConst, "singleconst"},
    {ScaleDualConst, "dualconst"},
    {ScaleDualCurve, "dualcurve"},
    {0, NULL}
};

static const Pair fm_pairs[] = {
    {FMNone, "none"},
    {FMDefault, "default"},
    {FMCluster, "cluster"},
    {FMCapi, "capi"},
    {0, NULL}
};

static const Pair fmstatus_pairs[] = {
    {FMStatusWholeWay, "wholeway"},
    {FMStatusHalfWay, "halfway"},
    {FMStatusFMFirst, "fmfirst"},
    {0, NULL}
};

template <typename TypeT>
static void parse_enum (const ContextParams &params, const Pair *pairs, const char *name, TypeT &value) {
    ContextParams::const_iterator iter = params.find (name);
    if (iter == params.end ())
        return;
    for (uint32_t i = 0; pairs[i].name != NULL; i++) {
        if (!strcasecmp (iter->second, pairs[i].name)) {
            value = (TypeT)pairs[i].id;
            break;
        }
    }
}

StitchContext::StitchContext ()
    : ContextBase (HandleTypeStitch)
    , _module (StitchSoft)
    , _cam_model (CamC3C8K)
    , _scopic_mode (ScopicStereoLeft)
    , _fisheye_num (3)
    , _blend_pyr_levels (1)
    , _scale_mode (ScaleSingleConst)
    , _dewarp_mode (DewarpSphere)
    , _fm_mode (FMDefault)
    , _fm_frames (120)
    , _fm_status (FMStatusWholeWay)
{
    xcam_mem_clear (_viewpoints_range);
}

StitchContext::~StitchContext ()
{
}

XCamReturn
StitchContext::set_parameters (ContextParams &param_list)
{
    uint32_t help = 0;
    parse_value (param_list, "help", help);
    if (help)
        show_help ();

    parse_enum (param_list, cammodel_pairs, "cammodel", _cam_model);
    parse_enum (param_list, scopic_pairs, "scopic", _scopic_mode);
    parse_enum (param_list, module_pairs, "module", _module);
    parse_enum (param_list, dewarp_pairs, "dewarp", _dewarp_mode);
    parse_enum (param_list, scale_pairs, "scale", _scale_mode);
    parse_enum (param_list, fm_pairs, "fm", _fm_mode);
    parse_enum (param_list, fmstatus_pairs, "fmstatus", _fm_status);
    parse_value (param_list, "fmframes", _fm_frames);
    parse_value (param_list, "fisheyenum", _fisheye_num);
    parse_value (param_list, "levels", _blend_pyr_levels);

    if (_module != StitchSoft) {
        set_alloc_out_buf (true);
        set_mem_type (XCAM_MEM_TYPE_GPU);
    }

    create_buf_pool (_module);

    ContextBase::set_parameters (param_list);
    show_options ();

    CamModel cam_model = (CamModel)_cam_model;
    viewpoints_range (cam_model, _viewpoints_range);
    _fm_cfg = (_module == StitchVulkan) ? vk_fm_config (cam_model) :
              ((_module == StitchGLES) ? gl_fm_config (cam_model) : soft_fm_config (cam_model));

    if (_dewarp_mode == DewarpSphere) {
        _fm_region_ratio = fm_region_ratio (cam_model);

        StitchScopicMode scopic_mode = (StitchScopicMode)_scopic_mode;
        _stich_info = (_module == StitchSoft) ?
                      soft_stitch_info (cam_model, scopic_mode) : gl_stitch_info (cam_model, scopic_mode);
    } else {
        _bowl_cfg = bowl_config (cam_model);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitchContext::init_handler ()
{
    SmartPtr<Stitcher> stitcher = create_stitcher (_module);
    XCAM_ASSERT (stitcher.ptr ());
    _stitcher = stitcher;

    init_config ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitchContext::uinit_handler ()
{
    if (_stitcher.ptr ())
        _stitcher.release ();

    return XCAM_RETURN_NO_ERROR;
}

bool
StitchContext::is_handler_valid () const
{
    return _stitcher.ptr () ? true : false;
}

XCamReturn
StitchContext::execute (SmartPtr<VideoBuffer> &buf_in, SmartPtr<VideoBuffer> &buf_out)
{
    XCAM_FAIL_RETURN (
        ERROR, buf_in.ptr () && (need_alloc_out_buf () || buf_out.ptr ()),
        XCAM_RETURN_ERROR_MEM, "input or output buffer is NULL");

    VideoBufferList in_buffers;
    in_buffers.push_back (buf_in);

    SmartPtr<VideoBuffer> pre_buf = buf_in;
    SmartPtr<VideoBuffer> att_buf = pre_buf->find_typed_attach<VideoBuffer> ();
    while (att_buf.ptr ()) {
        pre_buf->detach_buffer (att_buf);
        in_buffers.push_back (att_buf);

        pre_buf = att_buf;
        att_buf = pre_buf->find_typed_attach<VideoBuffer> ();
    }

    return _stitcher->stitch_buffers (in_buffers, buf_out);
}

SmartPtr<Stitcher>
StitchContext::create_stitcher (StitchModule module)
{
    SmartPtr<Stitcher> stitcher;

    if (module == StitchSoft) {
        stitcher = Stitcher::create_soft_stitcher ();
    } else if (module == StitchGLES) {
#if HAVE_GLES
        stitcher = Stitcher::create_gl_stitcher ();
#endif
    } else if (module == StitchVulkan) {
#if HAVE_VULKAN
        stitcher = Stitcher::create_vk_stitcher (VKDevice::default_device ());
#endif
    }
    XCAM_ASSERT (stitcher.ptr ());

    return stitcher;
}

XCamReturn
StitchContext::create_buf_pool (StitchModule module)
{
    SmartPtr<BufferPool> pool;
    if (module == StitchSoft) {
        pool = new SoftVideoBufAllocator ();
    } else if (module == StitchGLES) {
#if HAVE_GLES
        SmartPtr<EGLBase> egl = EGLBase::instance ();
        XCAM_ASSERT (egl.ptr ());

        XCAM_FAIL_RETURN (ERROR, egl->init (), XCAM_RETURN_ERROR_MEM, "init EGL failed");

        pool = new GLVideoBufferPool ();
#endif
    } else if (module == StitchVulkan) {
#if HAVE_VULKAN
        pool = create_vk_buffer_pool (VKDevice::default_device ());
        XCAM_ASSERT (pool.ptr ());
#endif
    }
    XCAM_ASSERT (pool.ptr ());

    set_buf_pool (pool);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitchContext::init_config ()
{
    XCAM_ASSERT (_stitcher.ptr ());

    _stitcher->set_camera_num (_fisheye_num);
    _stitcher->set_output_size (get_out_width (), get_out_height ());
    _stitcher->set_dewarp_mode (_dewarp_mode);
    _stitcher->set_scale_mode (_scale_mode);
    _stitcher->set_blend_pyr_levels (_blend_pyr_levels);
    _stitcher->set_fm_mode (_fm_mode);
#if HAVE_OPENCV
    _stitcher->set_fm_frames (_fm_frames);
    _stitcher->set_fm_status (_fm_status);
    _stitcher->set_fm_config (_fm_cfg);
#endif
    _stitcher->set_viewpoints_range (_viewpoints_range);

    if (_dewarp_mode == DewarpSphere) {
#if HAVE_OPENCV
        _stitcher->set_fm_region_ratio (_fm_region_ratio);
#endif
        get_fisheye_info ((CamModel)_cam_model, (StitchScopicMode)_scopic_mode, _stich_info.fisheye_info);

        for (uint32_t cam_id = 0; cam_id < XCAM_STITCH_FISHEYE_MAX_NUM; cam_id++) {
            XCAM_LOG_DEBUG ("cam[%d]: flip=%d ", cam_id, _stich_info.fisheye_info[cam_id].intrinsic.flip);
            XCAM_LOG_DEBUG ("fx=%f ", _stich_info.fisheye_info[cam_id].intrinsic.fx);
            XCAM_LOG_DEBUG ("fy=%f ", _stich_info.fisheye_info[cam_id].intrinsic.fy);
            XCAM_LOG_DEBUG ("cx=%f ", _stich_info.fisheye_info[cam_id].intrinsic.cx);
            XCAM_LOG_DEBUG ("cy=%f ", _stich_info.fisheye_info[cam_id].intrinsic.cy);
            XCAM_LOG_DEBUG ("w=%d ", _stich_info.fisheye_info[cam_id].intrinsic.width);
            XCAM_LOG_DEBUG ("h=%d ", _stich_info.fisheye_info[cam_id].intrinsic.height);
            XCAM_LOG_DEBUG ("fov=%f ", _stich_info.fisheye_info[cam_id].intrinsic.fov);
            XCAM_LOG_DEBUG ("skew=%f ", _stich_info.fisheye_info[cam_id].intrinsic.skew);
            XCAM_LOG_DEBUG ("radius=%f ", _stich_info.fisheye_info[cam_id].radius);
            XCAM_LOG_DEBUG ("distroy coeff=%f %f %f %f ", _stich_info.fisheye_info[cam_id].distort_coeff[0], _stich_info.fisheye_info[cam_id].distort_coeff[1], _stich_info.fisheye_info[cam_id].distort_coeff[2], _stich_info.fisheye_info[cam_id].distort_coeff[3]);
            XCAM_LOG_DEBUG ("fisheye eluer angles: yaw:%f, pitch:%f, roll:%f", _stich_info.fisheye_info[cam_id].extrinsic.yaw, _stich_info.fisheye_info[cam_id].extrinsic.pitch, _stich_info.fisheye_info[cam_id].extrinsic.roll);
            XCAM_LOG_DEBUG ("fisheye translation: x:%f, y:%f, z:%f", _stich_info.fisheye_info[cam_id].extrinsic.trans_x, _stich_info.fisheye_info[cam_id].extrinsic.trans_y, _stich_info.fisheye_info[cam_id].extrinsic.trans_z);
        }

        _stitcher->set_stitch_info (_stich_info);
    } else {
        _stitcher->set_intrinsic_names (intrinsic_names);
        _stitcher->set_extrinsic_names (extrinsic_names);
        _stitcher->set_bowl_config (_bowl_cfg);
    }

    return XCAM_RETURN_NO_ERROR;
}

void
StitchContext::show_help ()
{
    printf (
        "Usage:  params=help=1 module=soft fisheyenum=3 ...\n"
        "  module      : Processing module\n"
        "                Range   : [soft, gles, vulkan]\n"
        "                Default : soft\n"
        "  fisheyenum  : Number of fisheye lens\n"
        "                Range   : [2 - %d]\n"
        "                Default : 3\n"
        "  cammodel    : Camera model\n"
        "                Range   : [cama2c1080p, camb4c1080p, camc3c8k, camd3c8k]\n"
        "                Default : camc3c8k\n"
        "  levels      : The pyramid levels of blender\n"
        "                Range   : [1 - 4]\n"
        "                Default : 1\n"
        "  dewarp      : Fisheye dewarp mode\n"
        "                Range   : [sphere, bowl]\n"
        "                Default : sphere\n"
        "  scopic      : Scopic mode\n"
        "                Range   : [mono, stereoleft, stereoright]\n"
        "                Default : mono\n"
        "  scale       : Scaling mode for geometric mapping\n"
        "                Range   : [singleconst, dualconst, dualcurve]\n"
        "                Default : singleconst\n"
#if HAVE_OPENCV
        "  fm          : Feature match mode\n"
        "                Range   : [none, default, cluster, capi]\n"
        "                Default : default\n"
        "  fmframes    : How many frames need to run feature match at the beginning\n"
        "                Range   : [0 - INT_MAX]\n"
        "                Default : 120\n"
        "  fmstatus    : Running status of feature match\n"
        "                Range   : [fmfirst, halfway, wholeway]\n"
        "                Default : wholeway\n"
        "                  wholeway: run feature match during the entire runtime\n"
        "                  halfway : run feature match with stitching in the first fmframes frames\n"
        "                  fmfirst : run feature match without stitching in the first fmframes frames\n"
#else
        "  fm          : Feature match mode\n"
        "                Range   : [none]\n"
        "                Default : none\n"
#endif
        "  help        : Print usage\n"
        "                Range   : [0, 1]\n"
        "                Default : 0\n",
        XCAM_MAX_INPUTS_NUM);
}

void
StitchContext::show_options ()
{
    printf ("Options:\n");
    printf ("  Camera model\t\t: %s\n", cammodel_pairs[_cam_model].name);
    printf ("  Stitch module\t\t: %s\n", module_pairs[_module].name);
    printf ("  Input width\t\t: %d\n", get_in_width ());
    printf ("  Input height\t\t: %d\n", get_in_height ());
    printf ("  Output width\t\t: %d\n", get_out_width ());
    printf ("  Output height\t\t: %d\n", get_out_height ());
    printf ("  Pixel format\t\t: %s\n", get_format () == V4L2_PIX_FMT_YUV420 ? "yuv420" : "nv12");
    printf ("  Fisheye number\t: %d\n", _fisheye_num);
    printf ("  Blend pyr levels\t: %d\n", _blend_pyr_levels);
    printf ("  Dewarp mode\t\t: %s\n", dewarp_pairs[_dewarp_mode].name);
    printf ("  Scopic mode\t\t: %s\n", scopic_pairs[_scopic_mode].name);
    printf ("  Scaling mode\t\t: %s\n", scale_pairs[_scale_mode].name);
    printf ("  Feature match\t\t: %s\n", fm_pairs[_fm_mode].name);
#if HAVE_OPENCV
    printf ("  Feature match frames\t: %d\n", _fm_frames);
    printf ("  Feature match status\t: %s\n", fmstatus_pairs[_fm_status].name);
#endif
}

}

