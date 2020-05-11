/*
 * context_cl.cpp - private context for OpenCL modules
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "context_cl.h"
#include "ocl/cl_device.h"
#include "ocl/cl_utils.h"
#include "ocl/cl_newwavelet_denoise_handler.h"
#include "ocl/cl_defog_dcp_handler.h"
#include "ocl/cl_3d_denoise_handler.h"
#include "ocl/cl_image_warp_handler.h"
#include "ocl/cl_fisheye_handler.h"
#include "ocl/cl_image_360_stitch.h"

namespace XCam {

static const char *intrinsic_names[] = {
    "intrinsic_camera_front.txt",
    "intrinsic_camera_right.txt",
    "intrinsic_camera_rear.txt",
    "intrinsic_camera_left.txt"
};
static const char *extrinsic_names[] = {
    "extrinsic_camera_front.txt",
    "extrinsic_camera_right.txt",
    "extrinsic_camera_rear.txt",
    "extrinsic_camera_left.txt"
};

typedef struct Pair {
    uint32_t id;
    const char *name;
} Pair;

static const Pair dewarp_pairs[] = {
    {DewarpSphere, "sphere"},
    {DewarpBowl,   "bowl"},
    {0, NULL}
};

static const Pair res_pairs[] = {
    {StitchRes1080P2Cams, "1080p2cams"},
    {StitchRes1080P4Cams, "1080p4cams"},
    {StitchRes4K2Cams,    "4k2cams"},
    {StitchRes8K3Cams,    "8k3cams"},
    {StitchRes8K6Cams,    "8k6cams"},
    {0, NULL}
};

static const Pair scale_pairs[] = {
    {CLBlenderScaleLocal,  "local"},
    {CLBlenderScaleGlobal, "global"},
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

CLContextBase::CLContextBase (HandleType type)
    : ContextBase (type)
{
    if (!get_input_buffer_pool ().ptr()) {
        SmartPtr<BufferPool> pool = new CLVideoBufferPool ();
        XCAM_ASSERT (pool.ptr ());

        set_buf_pool (pool);
    }
}

CLContextBase::~CLContextBase ()
{
}

XCamReturn
CLContextBase::init_handler ()
{
    SmartPtr<CLContext> cl_context = CLDevice::instance()->get_context ();
    XCAM_FAIL_RETURN (
        ERROR, cl_context.ptr (), XCAM_RETURN_ERROR_UNKNOWN,
        "CLContextBase::init_handler(%s) failed since cl-context is NULL",
        get_type_name ());

    SmartPtr<CLImageHandler> handler = create_handler (cl_context);
    XCAM_FAIL_RETURN (
        ERROR, handler.ptr (), XCAM_RETURN_ERROR_UNKNOWN,
        "CLContextBase::init_handler(%s) create handler failed", get_type_name ());

    handler->disable_buf_pool (!need_alloc_out_buf ());
    _handler = handler;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLContextBase::uinit_handler ()
{
    if (!_handler.ptr ())
        return XCAM_RETURN_NO_ERROR;

    _handler->emit_stop ();
    _handler.release ();
    return XCAM_RETURN_NO_ERROR;
}

bool
CLContextBase::is_handler_valid () const
{
    return _handler.ptr () ? true : false;
}

XCamReturn
CLContextBase::execute (SmartPtr<VideoBuffer> &buf_in, SmartPtr<VideoBuffer> &buf_out)
{
    if (!need_alloc_out_buf ()) {
        XCAM_FAIL_RETURN (
            ERROR, buf_out.ptr (), XCAM_RETURN_ERROR_MEM,
            "context (%s) execute failed, buf_out need set.", get_type_name ());
    } else {
        XCAM_FAIL_RETURN (
            ERROR, !buf_out.ptr (), XCAM_RETURN_ERROR_MEM,
            "context (%s) execute failed, buf_out need NULL.", get_type_name ());
    }

    return _handler->execute (buf_in, buf_out);
}

SmartPtr<CLImageHandler>
NR3DContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_3d_denoise_image_handler (
               context, CL_IMAGE_CHANNEL_Y | CL_IMAGE_CHANNEL_UV, 3);
}

SmartPtr<CLImageHandler>
NRWaveletContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_newwavelet_denoise_image_handler (
               context, CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y, false);
}

FisheyeContext::FisheyeContext ()
    : CLContextBase (HandleTypeFisheye)
    , _range_longitude (228.0f)
    , _range_latitude (180.0f)
{
    _info.intrinsic.cx = 480.0f;
    _info.intrinsic.cy = 480.0f;
    _info.intrinsic.fov = 202.8f;
    _info.radius = 480.0f;
    _info.extrinsic.roll = -90.0f;
}

FisheyeContext::~FisheyeContext ()
{
}

SmartPtr<CLImageHandler>
FisheyeContext::create_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> handler = create_fisheye_handler (context);
    SmartPtr<CLFisheyeHandler> fisheye = handler.dynamic_cast_ptr<CLFisheyeHandler> ();
    XCAM_ASSERT (fisheye.ptr ());

    fisheye->set_fisheye_info (_info);
    fisheye->set_dst_range (_range_longitude, _range_latitude);
    fisheye->set_output_size (get_out_width (), get_out_height ());

    return handler;
}

SmartPtr<CLImageHandler>
DefogContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_defog_dcp_image_handler (context);;
}

SmartPtr<CLImageHandler>
DVSContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_image_warp_handler (context);
}

StitchCLContext::StitchCLContext ()
    : CLContextBase (HandleTypeStitchCL)
    , _enable_fisheyemap (false)
    , _enable_fm (true)
    , _enable_lsc (false)
    , _enable_seam (false)
    , _fisheye_num (2)
    , _dewarp_mode (DewarpSphere)
    , _res_mode (StitchRes1080P2Cams)
    , _scale_mode (CLBlenderScaleLocal)
{
}

StitchCLContext::~StitchCLContext ()
{
}

XCamReturn
StitchCLContext::set_parameters (ContextParams &param_list)
{
    uint32_t help = 0;
    parse_value (param_list, "help", help);
    if (help)
        show_help ();

    parse_enum (param_list, dewarp_pairs, "dewarp", _dewarp_mode);
    parse_enum (param_list, res_pairs, "res", _res_mode);
    parse_enum (param_list, scale_pairs, "scale", _scale_mode);
    parse_value (param_list, "fisheyemap", _enable_fisheyemap);
    parse_value (param_list, "fm", _enable_fm);
    parse_value (param_list, "lsc", _enable_lsc);
    parse_value (param_list, "seam", _enable_seam);
    parse_value (param_list, "fisheyenum", _fisheye_num);

    ContextBase::set_parameters (param_list);
    show_options ();

    return XCAM_RETURN_NO_ERROR;
}

void
StitchCLContext::show_help ()
{
    printf (
        "Usage:  params=help=1 res=1080p2cams dewarp=sphere ...\n"
        "  res         : Resolution mode\n"
        "                Range   : [1080p2cams, 1080p4cams, 4k2cams, 8k3cams, 8k6cams]\n"
        "                Default : 1080p2cams\n"
        "  dewarp      : Fisheye dewarp mode\n"
        "                Range   : [sphere, bowl]\n"
        "                Default : sphere\n"
        "  scale       : Scaling mode for geometric mapping\n"
        "                Range   : [local, global]\n"
        "                Default : local\n"
        "  fisheyenum  : Number of fisheye lens\n"
        "                Range   : [2 - %d]\n"
        "                Default : 2\n"
#if HAVE_OPENCV
        "  fm          : Enable feature match\n"
        "                Range   : [0, 1]\n"
        "                Default : 1\n"
#endif
        "  fisheyemap  : Enable fisheye map\n"
        "                Range   : [0, 1]\n"
        "                Default : 0\n"
        "  lsc         : Enable lens shading correction\n"
        "                Range   : [0, 1]\n"
        "                Default : 0\n"
        "  seam        : Enable seam finder in blending area\n"
        "                Range   : [0, 1]\n"
        "                Default : 0\n"
        "  help        : Printf usage\n"
        "                Range   : [0, 1]\n"
        "                Default : 0\n",
        XCAM_MAX_INPUTS_NUM);
}

void
StitchCLContext::show_options ()
{
    printf ("Options:\n");
    printf ("  Input width\t\t: %d\n", get_in_width ());
    printf ("  Input height\t\t: %d\n", get_in_height ());
    printf ("  Output width\t\t: %d\n", get_out_width ());
    printf ("  Output height\t\t: %d\n", get_out_height ());
    printf ("  Pixel format\t\t: %s\n", get_format () == V4L2_PIX_FMT_YUV420 ? "yuv420" : "nv12");
    printf ("  Alloc output buffer\t: %d\n", need_alloc_out_buf ());
    printf ("  Resolution mode\t: %s\n", res_pairs[_res_mode].name);
    printf ("  Dewarp mode\t\t: %s\n", dewarp_pairs[_dewarp_mode].name);
    printf ("  Scaling mode\t\t: %s\n", scale_pairs[_scale_mode].name);
    printf ("  Fisheye number\t: %d\n", _fisheye_num);
#if HAVE_OPENCV
    printf ("  Enable feature match\t: %d\n", _enable_fm);
#endif
    printf ("  Enable fisheye map\t: %d\n", _enable_fisheyemap);
    printf ("  Enable lsc\t\t: %d\n", _enable_lsc);
    printf ("  Enable seam\t\t: %d\n", _enable_seam);
}

SmartPtr<CLImageHandler>
StitchCLContext::create_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImage360Stitch> image_360 = create_image_360_stitch (context, _enable_seam, _scale_mode,
                                           _enable_fisheyemap, _enable_lsc, _dewarp_mode, _res_mode, _fisheye_num).dynamic_cast_ptr<CLImage360Stitch> ();
    XCAM_FAIL_RETURN (ERROR, image_360.ptr (), NULL, "create image stitch handler failed");

    image_360->set_output_size (get_out_width (), get_out_height ());
#if HAVE_OPENCV
    image_360->set_feature_match (_enable_fm);
#endif

    if (_dewarp_mode == DewarpBowl) {
        image_360->set_intrinsic_names (intrinsic_names);
        image_360->set_extrinsic_names (extrinsic_names);
    }

    return image_360;
}

}

