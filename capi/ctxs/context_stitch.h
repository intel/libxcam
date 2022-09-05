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

enum StitchModule {
    StitchNone    = 0,
    StitchSoft,
    StitchGLES,
    StitchVulkan
};

enum GPU_ID {
    RenderD128 = 0,
    RenderD129,
    Card0,
};

class StitchContext
    : public ContextBase
{
public:
    StitchContext ();
    virtual ~StitchContext ();

    virtual XCamReturn set_parameters (ContextParams &param_list);

    virtual XCamReturn init_handler ();
    virtual XCamReturn uinit_handler ();
    virtual bool is_handler_valid () const;

    virtual XCamReturn execute (SmartPtr<VideoBuffer> &buf_in, SmartPtr<VideoBuffer> &buf_out);

private:
    SmartPtr<Stitcher> create_stitcher (StitchModule module);
    XCamReturn create_buf_pool (StitchModule module);
    XCamReturn init_config ();

    void show_help ();
    void show_options ();

private:
    XCAM_DEAD_COPY (StitchContext);

private:
    SmartPtr<Stitcher>        _stitcher;

    StitchModule              _module;
    uint32_t                  _cam_model;
    uint32_t                  _scopic_mode;
    uint32_t                  _fisheye_num;
    uint32_t                  _blend_pyr_levels;
    GeoMapScaleMode           _scale_mode;
    FisheyeDewarpMode         _dewarp_mode;
    FeatureMatchMode          _fm_mode;
    uint32_t                  _fm_frames;
    FeatureMatchStatus        _fm_status;
    GPU_ID                    _gpu_id;

    FMConfig                  _fm_cfg;
    FMRegionRatio             _fm_region_ratio;
    StitchInfo                _stich_info;
    BowlDataConfig            _bowl_cfg;
    float                     _viewpoints_range[XCAM_STITCH_FISHEYE_MAX_NUM];
    const char                *_node_name;
};

}

#endif // XCAM_CONTEXT_STITCH_H