/*
 * test_sv_params.h - parameters for surround view
 *
 *  Copyright (c) 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "interface/stitcher.h"

#if HAVE_JSON
#include <calibration_parser.h>
#endif

namespace XCam {

enum CamModel {
    CamA2C1080P = 0,
    CamB4C1080P,
    CamC3C8K,
    CamD3C8K
};

enum StitchScopicMode {
    ScopicMono,
    ScopicStereoLeft,
    ScopicStereoRight
};

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

#if HAVE_JSON
static const char *camera_calibration_json_names[] = {
    "",
    "",
    "",
    "k_camera_calibration.json"
};
#endif

uint32_t *
get_fisheye_img_roi_radius (
    CamModel model, StitchScopicMode scopic_mode, uint32_t *roi_radius)
{
    // enable macro XCAM_FISHEYE_IMG_ROI_RADIUS, and fine-tune the roi_radius

    switch (model) {
    case CamA2C1080P: {
        roi_radius[0] = 456;
        roi_radius[1] = 456;
        break;
    }
    case CamC3C8K: {
        switch (scopic_mode) {
        case ScopicStereoLeft: {
            roi_radius[0] = 1787;
            roi_radius[1] = 1787;
            roi_radius[2] = 1787;
            break;
        }
        case ScopicStereoRight: {
            roi_radius[0] = 1787;
            roi_radius[1] = 1787;
            roi_radius[2] = 1787;
            break;
        }
        default:
            XCAM_LOG_ERROR ("unsupported scopic mode (%d)", scopic_mode);
            break;
        }
        break;
    }
    case CamD3C8K: {
        switch (scopic_mode) {
        case ScopicStereoLeft: {
            roi_radius[0] = 1802;
            roi_radius[1] = 1802;
            roi_radius[2] = 1802;
            break;
        }
        case ScopicStereoRight: {
            roi_radius[0] = 1801;
            roi_radius[1] = 1801;
            roi_radius[2] = 1801;
            break;
        }
        default:
            XCAM_LOG_ERROR ("unsupported scopic mode (%d)", scopic_mode);
            break;
        }
        break;
    }
    default:
        XCAM_LOG_ERROR ("unsupported camera model (%d)", model);
        break;
    }

    return roi_radius;
}

BowlDataConfig
bowl_config (CamModel model)
{
    BowlDataConfig bowl;

    switch (model) {
    case CamB4C1080P: {
        bowl.a = 6060.0f;
        bowl.b = 4388.0f;
        bowl.c = 3003.4f;
        bowl.angle_start = 0.0f;
        bowl.angle_end = 360.0f;
        bowl.center_z = 1500.0f;
        bowl.wall_height = 1800.0f;
        bowl.ground_length = 3000.0f;
        break;
    }
    default:
        XCAM_LOG_ERROR ("unsupported camera model (%d)", model);
        break;
    }

    return bowl;
}

float *
viewpoints_range (CamModel model, float *range)
{
    switch (model) {
    case CamA2C1080P: {
        range[0] = 202.8f;
        range[1] = 202.8f;
        break;
    }
    case CamB4C1080P: {
        range[0] = 64.0f;
        range[1] = 160.0f;
        range[2] = 64.0f;
        range[3] = 160.0f;
        break;
    }
    case CamC3C8K: {
        range[0] = 144.0f;
        range[1] = 144.0f;
        range[2] = 144.0f;
        break;
    }
    case CamD3C8K: {
        range[0] = 132.0f;
        range[1] = 132.0f;
        range[2] = 132.0f;
        break;
    }
    default:
        XCAM_LOG_ERROR ("unknown camera model (%d)", model);
        break;
    }

    return range;
}

FMRegionRatio
fm_region_ratio (CamModel model)
{
    FMRegionRatio ratio;

    switch (model) {
    case CamA2C1080P: {
        ratio.pos_x = 0.0f;
        ratio.width = 1.0f;
        ratio.pos_y = 1.0f / 3.0f;
        ratio.height = 1.0f / 3.0f;
        break;
    }
    case CamC3C8K: {
        ratio.pos_x = 0.0f;
        ratio.width = 1.0f;
        ratio.pos_y = 1.0f / 3.0f;
        ratio.height = 1.0f / 3.0f;
        break;
    }
    case CamD3C8K: {
        ratio.pos_x = 0.0f;
        ratio.width = 1.0f;
        ratio.pos_y = 1.0f / 3.0f;
        ratio.height = 1.0f / 3.0f;
        break;
    }
    default:
        XCAM_LOG_ERROR ("unsupported camera model (%d)", model);
        break;
    }

    return ratio;
}

FMConfig
soft_fm_config (CamModel model)
{
    FMConfig cfg;

    switch (model) {
    case CamA2C1080P: {
        cfg.stitch_min_width = 136;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.9f;
        cfg.delta_mean_offset = 120.0f;
        cfg.recur_offset_error = 8.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 8.0f;
        cfg.max_track_error = 28.0f;
        break;
    }
    case CamB4C1080P: {
        cfg.stitch_min_width = 136;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.8f;
        cfg.delta_mean_offset = 120.0f;
        cfg.recur_offset_error = 8.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 20.0f;
        cfg.max_track_error = 28.0f;
#ifdef ANDROID
        cfg.max_track_error = 3600.0f;
#endif
        break;
    }
    case CamC3C8K: {
        cfg.stitch_min_width = 136;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.95f;
        cfg.delta_mean_offset = 256.0f;
        cfg.recur_offset_error = 4.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 20.0f;
        cfg.max_track_error = 6.0f;
        break;
    }
    case CamD3C8K: {
        cfg.stitch_min_width = 256;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.6f;
        cfg.delta_mean_offset = 256.0f;
        cfg.recur_offset_error = 2.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 32.0f;
        cfg.max_track_error = 10.0f;
        break;
    }
    default:
        XCAM_LOG_ERROR ("unknown camera model (%d)", model);
        break;
    }

    return cfg;
}

XCamReturn
get_fisheye_info (CamModel model, StitchScopicMode scopic_mode, FisheyeInfo* fisheye_info)
{
    XCamReturn ret = XCAM_RETURN_BYPASS;

#if HAVE_JSON
    CalibrationParser parser;
    StitchInfo info;

    ret = parser.parse_fisheye_camera_param (camera_calibration_json_names[model], info.fisheye_info, XCAM_STITCH_FISHEYE_MAX_NUM);
    if (XCAM_RETURN_NO_ERROR != ret) {
        return ret;
    }

    switch (scopic_mode) {
    case ScopicStereoLeft: {
        for (uint32_t i = 0; i < 3; i++) {
            fisheye_info[i].intrinsic = info.fisheye_info[2 * i].intrinsic;
            fisheye_info[i].extrinsic = info.fisheye_info[2 * i].extrinsic;
            for (uint32_t j = 0; j < sizeof (FisheyeInfo::distort_coeff) / sizeof(float); j++) {
                fisheye_info[i].distort_coeff[j] = info.fisheye_info[2 * i].distort_coeff[j];
            }
        }
        break;
    }
    case ScopicStereoRight: {
        for (uint32_t i = 0; i < 3; i++) {
            fisheye_info[i].intrinsic = info.fisheye_info[2 * i + 1].intrinsic;
            fisheye_info[i].extrinsic = info.fisheye_info[2 * i + 1].extrinsic;
            for (uint32_t j = 0; j < sizeof (FisheyeInfo::distort_coeff) / sizeof(float); j++) {
                fisheye_info[i].distort_coeff[j] = info.fisheye_info[2 * i + 1].distort_coeff[j];
            }
        }
        break;
    }
    default: {
        XCAM_LOG_ERROR ("unsupported scopic mode (%d)", scopic_mode);
        break;
    }
    }
#endif
    return ret;
}

StitchInfo
soft_stitch_info (CamModel model, StitchScopicMode scopic_mode)
{
    StitchInfo info;

    switch (model) {
    case CamA2C1080P: {
        info.fisheye_info[0].intrinsic.cx = 480.0f;
        info.fisheye_info[0].intrinsic.cy = 480.0f;
        info.fisheye_info[0].intrinsic.fov = 202.8f;
        info.fisheye_info[0].radius = 480.0f;
        info.fisheye_info[0].extrinsic.roll = -90.0f;
        info.fisheye_info[1].intrinsic.cx = 1436.0f;
        info.fisheye_info[1].intrinsic.cy = 480.0f;
        info.fisheye_info[1].intrinsic.fov = 202.8f;
        info.fisheye_info[1].radius = 480.0f;
        info.fisheye_info[1].extrinsic.roll = 89.7f;
        break;
    }
    case CamC3C8K: {
        switch (scopic_mode) {
        case ScopicStereoLeft: {
            info.merge_width[0] = 256;
            info.merge_width[1] = 256;
            info.merge_width[2] = 256;

            info.fisheye_info[0].intrinsic.cx = 1907.0f;
            info.fisheye_info[0].intrinsic.cy = 1440.0f;
            info.fisheye_info[0].intrinsic.fov = 200.0f;
            info.fisheye_info[0].radius = 1984.0f;
            info.fisheye_info[0].extrinsic.roll = 90.3f;
            info.fisheye_info[1].intrinsic.cx = 1920.0f;
            info.fisheye_info[1].intrinsic.cy = 1440.0f;
            info.fisheye_info[1].intrinsic.fov = 200.0f;
            info.fisheye_info[1].radius = 1984.0f;
            info.fisheye_info[1].extrinsic.roll = 90.2f;
            info.fisheye_info[2].intrinsic.cx = 1920.0f;
            info.fisheye_info[2].intrinsic.cy = 1440.0f;
            info.fisheye_info[2].intrinsic.fov = 200.0f;
            info.fisheye_info[2].radius = 1984.0f;
            info.fisheye_info[2].extrinsic.roll = 91.2f;
            break;
        }
        case ScopicStereoRight: {
            info.merge_width[0] = 256;
            info.merge_width[1] = 256;
            info.merge_width[2] = 256;

            info.fisheye_info[0].intrinsic.cx = 1920.0f;
            info.fisheye_info[0].intrinsic.cy = 1440.0f;
            info.fisheye_info[0].intrinsic.fov = 200.0f;
            info.fisheye_info[0].radius = 1984.0f;
            info.fisheye_info[0].extrinsic.roll = 90.0f;
            info.fisheye_info[1].intrinsic.cx = 1920.0f;
            info.fisheye_info[1].intrinsic.cy = 1440.0f;
            info.fisheye_info[1].intrinsic.fov = 200.0f;
            info.fisheye_info[1].radius = 1984.0f;
            info.fisheye_info[1].extrinsic.roll = 90.0f;
            info.fisheye_info[2].intrinsic.cx = 1914.0f;
            info.fisheye_info[2].intrinsic.cy = 1440.0f;
            info.fisheye_info[2].intrinsic.fov = 200.0f;
            info.fisheye_info[2].radius = 1984.0f;
            info.fisheye_info[2].extrinsic.roll = 90.1f;
            break;
        }
        default:
            XCAM_LOG_ERROR ("unsupported scopic mode (%d)", scopic_mode);
            break;
        }
        break;
    }
    case CamD3C8K: {
        switch (scopic_mode) {
        case ScopicStereoLeft: {
            info.merge_width[0] = 192;
            info.merge_width[1] = 192;
            info.merge_width[2] = 192;
            info.fisheye_info[0].intrinsic.cx = 1804.0f;
            info.fisheye_info[0].intrinsic.cy = 1532.0f;
            info.fisheye_info[0].intrinsic.fov = 190.0f;
            info.fisheye_info[0].radius = 1900.0f;
            info.fisheye_info[0].extrinsic.roll = 91.5f;
            info.fisheye_info[1].intrinsic.cx = 1836.0f;
            info.fisheye_info[1].intrinsic.cy = 1532.0f;
            info.fisheye_info[1].intrinsic.fov = 190.0f;
            info.fisheye_info[1].radius = 1900.0f;
            info.fisheye_info[1].extrinsic.roll = 92.0f;
            info.fisheye_info[2].intrinsic.cx = 1820.0f;
            info.fisheye_info[2].intrinsic.cy = 1532.0f;
            info.fisheye_info[2].intrinsic.fov = 190.0f;
            info.fisheye_info[2].radius = 1900.0f;
            info.fisheye_info[2].extrinsic.roll = 91.0f;
            break;
        }
        case ScopicStereoRight: {
            info.merge_width[0] = 192;
            info.merge_width[1] = 192;
            info.merge_width[2] = 192;
            info.fisheye_info[0].intrinsic.cx = 1836.0f;
            info.fisheye_info[0].intrinsic.cy = 1532.0f;
            info.fisheye_info[0].intrinsic.fov = 190.0f;
            info.fisheye_info[0].radius = 1900.0f;
            info.fisheye_info[0].extrinsic.roll = 88.0f;
            info.fisheye_info[1].intrinsic.cx = 1852.0f;
            info.fisheye_info[1].intrinsic.cy = 1576.0f;
            info.fisheye_info[1].intrinsic.fov = 190.0f;
            info.fisheye_info[1].radius = 1900.0f;
            info.fisheye_info[1].extrinsic.roll = 90.0f;
            info.fisheye_info[2].intrinsic.cx = 1836.0f;
            info.fisheye_info[2].intrinsic.cy = 1532.0f;
            info.fisheye_info[2].intrinsic.fov = 190.0f;
            info.fisheye_info[2].radius = 1900.0f;
            info.fisheye_info[2].extrinsic.roll = 91.0f;
            break;
        }
        default:
            XCAM_LOG_ERROR ("unsupported scopic mode (%d)", scopic_mode);
            break;
        }
        break;
    }
    default:
        XCAM_LOG_ERROR ("unsupported camera model (%d)", model);
        break;
    }

    return info;
}

FMConfig
gl_fm_config (CamModel model)
{
    FMConfig cfg;

    switch (model) {
    case CamA2C1080P: {
        cfg.stitch_min_width = 136;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.9f;
        cfg.delta_mean_offset = 120.0f;
        cfg.recur_offset_error = 8.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 8.0f;
        cfg.max_track_error = 28.0f;
        break;
    }
    case CamB4C1080P: {
        cfg.stitch_min_width = 136;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.8f;
        cfg.delta_mean_offset = 120.0f;
        cfg.recur_offset_error = 8.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 20.0f;
        cfg.max_track_error = 28.0f;
#ifdef ANDROID
        cfg.max_track_error = 3600.0f;
#endif
        break;
    }
    case CamC3C8K: {
        cfg.stitch_min_width = 136;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.95f;
        cfg.delta_mean_offset = 256.0f;
        cfg.recur_offset_error = 4.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 20.0f;
        cfg.max_track_error = 6.0f;
        break;
    }
    case CamD3C8K: {
        cfg.stitch_min_width = 256;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.6f;
        cfg.delta_mean_offset = 256.0f;
        cfg.recur_offset_error = 2.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 32.0f;
        cfg.max_track_error = 10.0f;
        break;
    }
    default:
        XCAM_LOG_ERROR ("unknown camera model (%d)", model);
        break;
    }

    return cfg;
}

StitchInfo
gl_stitch_info (CamModel model, StitchScopicMode scopic_mode)
{
    StitchInfo info;

    switch (model) {
    case CamA2C1080P: {
        info.fisheye_info[0].intrinsic.cx = 480.0f;
        info.fisheye_info[0].intrinsic.cy = 480.0f;
        info.fisheye_info[0].intrinsic.fov = 202.8f;
        info.fisheye_info[0].radius = 480.0f;
        info.fisheye_info[0].extrinsic.roll = -90.0f;
        info.fisheye_info[1].intrinsic.cx = 1436.0f;
        info.fisheye_info[1].intrinsic.cy = 480.0f;
        info.fisheye_info[1].intrinsic.fov = 202.8f;
        info.fisheye_info[1].radius = 480.0f;
        info.fisheye_info[1].extrinsic.roll = 89.7f;
        break;
    }
    case CamC3C8K: {
        switch (scopic_mode) {
        case ScopicStereoLeft: {
            info.merge_width[0] = 256;
            info.merge_width[1] = 256;
            info.merge_width[2] = 256;

            info.fisheye_info[0].intrinsic.cx = 1907.0f;
            info.fisheye_info[0].intrinsic.cy = 1440.0f;
            info.fisheye_info[0].intrinsic.fov = 200.0f;
            info.fisheye_info[0].radius = 1984.0f;
            info.fisheye_info[0].extrinsic.roll = 90.3f;
            info.fisheye_info[1].intrinsic.cx = 1920.0f;
            info.fisheye_info[1].intrinsic.cy = 1440.0f;
            info.fisheye_info[1].intrinsic.fov = 200.0f;
            info.fisheye_info[1].radius = 1984.0f;
            info.fisheye_info[1].extrinsic.roll = 90.2f;
            info.fisheye_info[2].intrinsic.cx = 1920.0f;
            info.fisheye_info[2].intrinsic.cy = 1440.0f;
            info.fisheye_info[2].intrinsic.fov = 200.0f;
            info.fisheye_info[2].radius = 1984.0f;
            info.fisheye_info[2].extrinsic.roll = 91.2f;
            break;
        }
        case ScopicStereoRight: {
            info.merge_width[0] = 256;
            info.merge_width[1] = 256;
            info.merge_width[2] = 256;

            info.fisheye_info[0].intrinsic.cx = 1920.0f;
            info.fisheye_info[0].intrinsic.cy = 1440.0f;
            info.fisheye_info[0].intrinsic.fov = 200.0f;
            info.fisheye_info[0].radius = 1984.0f;
            info.fisheye_info[0].extrinsic.roll = 90.0f;
            info.fisheye_info[1].intrinsic.cx = 1920.0f;
            info.fisheye_info[1].intrinsic.cy = 1440.0f;
            info.fisheye_info[1].intrinsic.fov = 200.0f;
            info.fisheye_info[1].radius = 1984.0f;
            info.fisheye_info[1].extrinsic.roll = 90.0f;
            info.fisheye_info[2].intrinsic.cx = 1914.0f;
            info.fisheye_info[2].intrinsic.cy = 1440.0f;
            info.fisheye_info[2].intrinsic.fov = 200.0f;
            info.fisheye_info[2].radius = 1984.0f;
            info.fisheye_info[2].extrinsic.roll = 90.1f;
            break;
        }
        default:
            XCAM_LOG_ERROR ("unsupported scopic mode (%d)", scopic_mode);
            break;
        }
        break;
    }
    case CamD3C8K: {
        switch (scopic_mode) {
        case ScopicStereoLeft: {
            info.merge_width[0] = 192;
            info.merge_width[1] = 192;
            info.merge_width[2] = 192;
            info.fisheye_info[0].intrinsic.cx = 1804.0f;
            info.fisheye_info[0].intrinsic.cy = 1532.0f;
            info.fisheye_info[0].intrinsic.fov = 190.0f;
            info.fisheye_info[0].radius = 1900.0f;
            info.fisheye_info[0].extrinsic.roll = 91.5f;
            info.fisheye_info[1].intrinsic.cx = 1836.0f;
            info.fisheye_info[1].intrinsic.cy = 1532.0f;
            info.fisheye_info[1].intrinsic.fov = 190.0f;
            info.fisheye_info[1].radius = 1900.0f;
            info.fisheye_info[1].extrinsic.roll = 92.0f;
            info.fisheye_info[2].intrinsic.cx = 1820.0f;
            info.fisheye_info[2].intrinsic.cy = 1532.0f;
            info.fisheye_info[2].intrinsic.fov = 190.0f;
            info.fisheye_info[2].radius = 1900.0f;
            info.fisheye_info[2].extrinsic.roll = 91.0f;
            break;
        }
        case ScopicStereoRight: {
            info.merge_width[0] = 192;
            info.merge_width[1] = 192;
            info.merge_width[2] = 192;
            info.fisheye_info[0].intrinsic.cx = 1836.0f;
            info.fisheye_info[0].intrinsic.cy = 1532.0f;
            info.fisheye_info[0].intrinsic.fov = 190.0f;
            info.fisheye_info[0].radius = 1900.0f;
            info.fisheye_info[0].extrinsic.roll = 88.0f;
            info.fisheye_info[1].intrinsic.cx = 1852.0f;
            info.fisheye_info[1].intrinsic.cy = 1576.0f;
            info.fisheye_info[1].intrinsic.fov = 190.0f;
            info.fisheye_info[1].radius = 1900.0f;
            info.fisheye_info[1].extrinsic.roll = 90.0f;
            info.fisheye_info[2].intrinsic.cx = 1836.0f;
            info.fisheye_info[2].intrinsic.cy = 1532.0f;
            info.fisheye_info[2].intrinsic.fov = 190.0f;
            info.fisheye_info[2].radius = 1900.0f;
            info.fisheye_info[2].extrinsic.roll = 91.0f;
            break;
        }
        default:
            XCAM_LOG_ERROR ("unsupported scopic mode (%d)", scopic_mode);
            break;
        }
        break;
    }
    default:
        XCAM_LOG_ERROR ("unsupported camera model (%d)", model);
        break;
    }

    return info;
}

FMConfig
vk_fm_config (CamModel model)
{
    FMConfig cfg;

    switch (model) {
    case CamB4C1080P: {
        cfg.stitch_min_width = 136;
        cfg.min_corners = 4;
        cfg.offset_factor = 0.8f;
        cfg.delta_mean_offset = 120.0f;
        cfg.recur_offset_error = 8.0f;
        cfg.max_adjusted_offset = 24.0f;
        cfg.max_valid_offset_y = 20.0f;
        cfg.max_track_error = 28.0f;
#ifdef ANDROID
        cfg.max_track_error = 3600.0f;
#endif
        break;
    }
    default:
        XCAM_LOG_ERROR ("unsupported camera model (%d)", model);
        break;
    }

    return cfg;
}

}

