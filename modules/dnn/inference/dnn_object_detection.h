/*
 * dnn_object_detection.h -  object detection
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
 * Author: Ali Mansouri <ali.m.t1992@gmail.com>
 */

#ifndef XCAM_DNN_OBJECT_DETECTION_H
#define XCAM_DNN_OBJECT_DETECTION_H

#pragma once

#include <string>

#include <xcam_std.h>
#include <vec_mat.h>
#include "dnn_inference_engine.h"

namespace XCam {

class DnnObjectDetection
    : public DnnInferenceEngine
{
public:
    explicit DnnObjectDetection (DnnInferConfig& config);
    virtual ~DnnObjectDetection ();

    virtual XCamReturn set_model_input_info (DnnInferInputOutputInfo& info);
    virtual XCamReturn get_model_input_info (DnnInferInputOutputInfo& info);

    virtual XCamReturn set_model_output_info (DnnInferInputOutputInfo& info);
    virtual XCamReturn get_model_output_info (DnnInferInputOutputInfo& info);

    XCamReturn get_bounding_boxes (const std::vector<float*> result_ptr,
                                   const uint32_t idx,
                                   std::vector<Vec4i> &boxes,
                                   std::vector<int32_t> &classes);

protected:
    virtual XCamReturn set_output_layer_type (const char* type);
};

}  // namespace XCam

#endif // XCAM_DNN_OBJECT_DETECTION_H

