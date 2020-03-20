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
#include "soft/soft_video_buf_allocator.h"
#if HAVE_GLES
#include "gles/gl_video_buffer.h"
#endif
#if HAVE_VULKAN
#include "vulkan/vk_device.h"
#endif

namespace XCam {

static void
init_default_params (FMConfig &cfg, FMRegionRatio &ratio, StitchInfo &info, float *range)
{
    cfg.stitch_min_width = 256;
    cfg.min_corners = 4;
    cfg.offset_factor = 0.6f;
    cfg.delta_mean_offset = 256.0f;
    cfg.recur_offset_error = 2.0f;
    cfg.max_adjusted_offset = 24.0f;
    cfg.max_valid_offset_y = 32.0f;
    cfg.max_track_error = 10.0f;

    ratio.pos_x = 0.0f;
    ratio.width = 1.0f;
    ratio.pos_y = 1.0f / 3.0f;
    ratio.height = 1.0f / 3.0f;

    range[0] =  154.0f;
    range[1] =  154.0f;
    range[2] =  154.0f;

    info.merge_width[0] = 192;
    info.merge_width[1] = 192;
    info.merge_width[2] = 192;
    info.fisheye_info[0].center_x = 1804.0f;
    info.fisheye_info[0].center_y = 1532.0f;
    info.fisheye_info[0].wide_angle = 190.0f;
    info.fisheye_info[0].radius = 1900.0f;
    info.fisheye_info[0].rotate_angle = 91.5f;
    info.fisheye_info[1].center_x = 1836.0f;
    info.fisheye_info[1].center_y = 1532.0f;
    info.fisheye_info[1].wide_angle = 190.0f;
    info.fisheye_info[1].radius = 1900.0f;
    info.fisheye_info[1].rotate_angle = 92.0f;
    info.fisheye_info[2].center_x = 1820.0f;
    info.fisheye_info[2].center_y = 1532.0f;
    info.fisheye_info[2].wide_angle = 190.0f;
    info.fisheye_info[2].radius = 1900.0f;
    info.fisheye_info[2].rotate_angle = 91.0f;
}

StitchContext::StitchContext ()
    : ContextBase (HandleTypeStitch)
    , _input_width (3840)
    , _input_height (2880)
    , _output_width (7680)
    , _output_height (3840)
    , _fisheye_num (3)
    , _module (StitchSoft)
    , _scale_mode (ScaleDualConst)
    , _blend_pyr_levels (1)
    , _fm_mode (FMCluster)
    , _dewarp_mode (DewarpSphere)
    , _fm_frames (120)
    , _fm_status (FMStatusFMFirst)

{
    xcam_mem_clear (_viewpoints_range);
    init_default_params (_fm_cfg, _fm_region_ratio, _stich_info, _viewpoints_range);

    create_buf_pool (V4L2_PIX_FMT_NV12);
}

StitchContext::~StitchContext ()
{
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
    if (!need_alloc_out_buf ()) {
        XCAM_FAIL_RETURN (ERROR, buf_out.ptr (), XCAM_RETURN_ERROR_MEM, "output buffer is NULL");
    } else {
        XCAM_FAIL_RETURN (ERROR, !buf_out.ptr (), XCAM_RETURN_ERROR_MEM, "output buffer need NULL");
    }

    VideoBufferList in_buffers;
    in_buffers.push_back (buf_in);

    SmartPtr<VideoBuffer> pre_buf;
    SmartPtr<VideoBuffer> cur_buf = buf_in;
    for (uint32_t i = 0; i < _fisheye_num; i++) {
        pre_buf = cur_buf;
        cur_buf = cur_buf->find_typed_attach<VideoBuffer> ();

        if (!cur_buf.ptr ()) {
            XCAM_FAIL_RETURN (
                ERROR, i == (_fisheye_num - 1), XCAM_RETURN_ERROR_PARAM,
                "conflicting attached buffers and fisheye number");
            break;
        }

        pre_buf->detach_buffer (cur_buf);
        in_buffers.push_back (cur_buf);
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
StitchContext::create_buf_pool (uint32_t format)
{
    VideoBufferInfo info;
    info.init (format, _input_width, _input_height);

    SmartPtr<BufferPool> pool;
    if (_module == StitchSoft) {
        pool = new SoftVideoBufAllocator (info);
    } else if (_module == StitchGLES) {
#if HAVE_GLES
        pool = new GLVideoBufferPool (info);
#endif
    } else if (_module == StitchVulkan) {
#if HAVE_VULKAN
        pool = create_vk_buffer_pool (VKDevice::default_device ());
        XCAM_ASSERT (pool.ptr ());
        pool->set_video_info (info);
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
    _stitcher->set_output_size (_output_width, _output_height);
    _stitcher->set_dewarp_mode (_dewarp_mode);
    _stitcher->set_scale_mode (_scale_mode);
    _stitcher->set_blend_pyr_levels (_blend_pyr_levels);
    _stitcher->set_fm_mode (_fm_mode);
#if HAVE_OPENCV
    _stitcher->set_fm_frames (_fm_frames);
    _stitcher->set_fm_status (_fm_status);
    _stitcher->set_fm_config (_fm_cfg);
    _stitcher->set_fm_region_ratio (_fm_region_ratio);
#endif
    _stitcher->set_viewpoints_range (_viewpoints_range);
    _stitcher->set_stitch_info (_stich_info);

    return XCAM_RETURN_NO_ERROR;
}

}

