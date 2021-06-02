/*
 * dnn_inference_utils.h -  dnn inference utils header file
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

#ifndef XCAM_DNN_INFERENCE_UTILS_H
#define XCAM_DNN_INFERENCE_UTILS_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <vec_mat.h>

#include "dnn_inference_engine.h"

namespace XCamDNN {

XCamReturn
draw_bounding_boxes (
    uint8_t *data,  uint32_t width, uint32_t height, XCam::DnnInferImageFormatType format,
    std::vector<XCam::Vec4i> rectangles, std::vector<int32_t> classes, int32_t thickness = 3);

XCamReturn
label_pixels (const std::string name, std::vector<std::vector<uint32_t>> map);

XCamReturn
save_bmp_file (const std::string name,
               void* data,
               XCam::DnnInferImageFormatType format,
               XCam::DnnInferPrecisionType precision,
               uint32_t width,
               uint32_t height);

//std::shared_ptr<uint8_t>
uint8_t*
convert_NV12_to_BGR (XCam::SmartPtr<XCam::VideoBuffer>& nv12, float x_ratio, float y_ratio);

uint8_t*
resize_BGR (XCam::SmartPtr<XCam::VideoBuffer>& bgr, float x_ratio, float y_ratio);

}  // namespace XCamDNN

#endif  //XCAM_DNN_INFERENCE_UTILS_H
