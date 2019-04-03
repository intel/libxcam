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

#include <vec_mat.h>

namespace XCamDNN {

XCamReturn
draw_bounding_boxes (
    uint8_t *data,  uint32_t width, uint32_t height,
    std::vector<XCam::Vec4i> rectangles, std::vector<int32_t> classes, int32_t thickness = 1);

XCamReturn
save_bmp_file (const std::string name, unsigned char* data, uint32_t width, uint32_t height);

}  // namespace XCamDNN

#endif  //XCAM_DNN_INFERENCE_UTILS_H
