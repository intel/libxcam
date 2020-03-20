/*
 * context_stitch.h - private context for image stitching
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

#ifndef XCAM_CONTEXT_STITCH_H
#define XCAM_CONTEXT_STITCH_H

#include <string.h>
#include "xcam_utils.h"
#include "context_priv.h"
#include "interface/stitcher.h"

namespace XCam {

enum SVModule {
    SVModuleNone    = 0,
    SVModuleSoft,
    SVModuleGLES,
    SVModuleVulkan
};

class SVContextBase
    : public ContextBase
{
public:
    SVContextBase ();
    virtual ~SVContextBase ();

    virtual XCamReturn init_handler ();
    virtual XCamReturn uinit_handler ();
    virtual bool is_handler_valid () const;

    virtual XCamReturn execute (SmartPtr<VideoBuffer> &buf_in, SmartPtr<VideoBuffer> &buf_out);

private:
    SmartPtr<Stitcher> create_stitcher (SVModule module);
    XCamReturn create_buf_pool (uint32_t format);
    XCamReturn init_config ();

private:
    XCAM_DEAD_COPY (SVContextBase);

private:
    SmartPtr<Stitcher>        _stitcher;

    uint32_t                  _input_width;
    uint32_t                  _input_height;
    uint32_t                  _output_width;
    uint32_t                  _output_height;

    uint32_t                  _fisheye_num;
    SVModule                  _module;
    GeoMapScaleMode           _scale_mode;

    uint32_t                  _blend_pyr_levels;

    FeatureMatchMode          _fm_mode;
    FisheyeDewarpMode         _dewarp_mode;
    uint32_t                  _fm_frames;
    FeatureMatchStatus        _fm_status;

    FMConfig                  _fm_cfg;
    FMRegionRatio             _fm_region_ratio;
    StitchInfo                _stich_info;
    float                     _viewpoints_range[XCAM_STITCH_FISHEYE_MAX_NUM];
};

}

#endif // XCAM_CONTEXT_STITCH_H