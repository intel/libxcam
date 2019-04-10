/*
 * dnn_semantic_segmentation.h -  semantic segmentation
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

#ifndef XCAM_DNN_SEMANTIC_SEGMENTATION_H
#define XCAM_DNN_SEMANTIC_SEGMENTATION_H

#pragma once

#include <string>

#include <xcam_std.h>
#include "dnn_inference_engine.h"

namespace XCam {

class DnnSemanticSegmentation
    : public DnnInferenceEngine
{
public:
    explicit DnnSemanticSegmentation (DnnInferConfig& config);
    virtual ~DnnSemanticSegmentation ();

    XCamReturn set_model_input_info (DnnInferInputOutputInfo& info);
    XCamReturn get_model_input_info (DnnInferInputOutputInfo& info);

    XCamReturn set_model_output_info (DnnInferInputOutputInfo& info);
    XCamReturn get_model_output_info (DnnInferInputOutputInfo& info);

    XCamReturn get_segmentation_map (const float* result_ptr,
                                     const uint32_t idx,
                                     std::vector<std::vector<uint32_t>>& out_classes);

protected:
    XCamReturn set_output_layer_type (const char* type);
};

}  // namespace XCam

#endif //XCAM_DNN_SEMANTIC_SEGMENTATION_H
