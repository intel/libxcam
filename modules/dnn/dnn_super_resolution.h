/*
 * dnn_super_resolution.h -  super resolution
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef _XCMA_DNN_SUPER_RESOLUTION_H_
#define _XCMA_DNN_SUPER_RESOLUTION_H_

#pragma once

#include <string>

#include <xcam_std.h>
#include "dnn_inference_engine.h"

namespace XCam {

class DnnSuperResolution
    : public DnnInferenceEngine
{
public:
    explicit DnnSuperResolution (DnnInferConfig& config);
    virtual ~DnnSuperResolution ();

    XCamReturn set_model_input_info (DnnInferInputOutputInfo& info);
    XCamReturn get_model_input_info (DnnInferInputOutputInfo& info);

    XCamReturn set_model_output_info (DnnInferInputOutputInfo& info);
    XCamReturn get_model_output_info (DnnInferInputOutputInfo& info);

    void* get_inference_results (uint32_t idx, uint32_t& size);
};

}  // namespace XCam

#endif //_XCMA_DNN_SUPER_RESOLUTION_H_
