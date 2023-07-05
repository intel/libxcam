/*
 * dnn_object_detection.cpp -  object detection
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

#include <openvino/openvino.hpp>

#include "dnn_object_detection.h"

using namespace std;
using namespace ov;

namespace XCam {

DnnObjectDetection::DnnObjectDetection (DnnInferConfig& config)
    : DnnInferenceEngine (config)
{
    XCAM_LOG_DEBUG ("DnnObjectDetection::DnnObjectDetection");
    set_output_layer_type ("DetectionOutput");
}

DnnObjectDetection::~DnnObjectDetection ()
{

}

XCamReturn
DnnObjectDetection::set_output_layer_type (const char* type)
{
    _output_layer_type.insert (DnnOutputLayerType::value_type (DnnInferObjectDetection, type));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnObjectDetection::get_model_input_info (DnnInferInputOutputInfo& info)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    for (size_t id = 0; id < get_input_size (); id ++) {
        if (_network->input (id).get_shape ().size() == 4) {
            XCAM_LOG_DEBUG ("Batch size is: %d", _network->input (id).get_shape ()[0]);
            info.width[id] = _network->input (id).get_shape ()[3];
            info.height[id] = _network->input (id).get_shape ()[2];
            info.channels[id] = _network->input (id).get_shape ()[1];
            info.object_size[id] = _network->input (id).get_shape ()[0];
            info.data_type[id] = DnnInferDataTypeImage;
            info.precision[id] = DnnInferPrecisionU8;
            info.layout[id] = DnnInferLayoutNCHW;
        } else if (_network->input (id).get_shape ().size() == 2) {
            info.precision[id] = DnnInferPrecisionFP32;
            if ((_network->input (id).get_shape ()[1] != 3 && _network->input (id).get_shape ()[1] != 6)) {
                XCAM_LOG_ERROR ("Invalid input info. Should be 3 or 6 values length");
                return XCAM_RETURN_ERROR_PARAM;
            }
        }
    }

    info.batch_size = _network->input (0).get_shape ()[0];
    info.numbers = get_input_size ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnObjectDetection::set_model_input_info (DnnInferInputOutputInfo& info)
{
    XCAM_LOG_DEBUG ("DnnObjectDetection::set_model_input_info");

    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (info.numbers != get_input_size ()) {
        XCAM_LOG_ERROR ("Input size is not matched with model info numbers %d !", info.numbers);
        return XCAM_RETURN_ERROR_PARAM;
    }

    ov::preprocess::PrePostProcessor ppp (_network);

    for (size_t idx = 0; idx < get_input_size (); idx ++) {
        ov::preprocess::InputInfo& input_info = ppp.input (idx);
        ov::element::Type precision = convert_precision_type (info.precision[idx]);
        ov::Layout layout = convert_layout_type (info.layout[idx]);

        input_info.tensor().set_element_type (precision);
        input_info.tensor().set_layout (layout);
        input_info.model().set_layout (layout);
    }

    _network = ppp.build();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnObjectDetection::get_model_output_info (DnnInferInputOutputInfo& info)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    std::string output_name;
    for (size_t idx = 0; idx < get_output_size (); idx ++) {
        if (output_name.empty ()) {
            output_name = *(_network->output(idx).get_names ().begin ());
        }

        const ov::Shape output_dims = _network->output (idx).get_shape ();

        info.width[idx]    = output_dims[0];
        info.height[idx]   = output_dims[1];
        info.channels[idx] = output_dims[2];
        info.object_size[idx] = output_dims[3];
        info.precision[idx] = DnnInferPrecisionFP32;
        info.layout[idx] = DnnInferLayoutNHWC;
        info.data_type[idx] = DnnInferDataTypeNonImage;
        info.format[idx] = DnnInferImageFormatUnknown;
    }

    info.batch_size = _network->output (0).get_shape ()[0];
    info.numbers = get_output_size ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnObjectDetection::set_model_output_info (DnnInferInputOutputInfo& info)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (info.numbers != get_output_size ()) {
        XCAM_LOG_ERROR ("Output size is not matched with model!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ov::preprocess::PrePostProcessor ppp (_network);

    for (size_t idx = 0; idx < get_output_size (); idx ++) {
        ov::preprocess::OutputInfo& output_info = ppp.output (idx);
        ov::element::Type precision = convert_precision_type (info.precision[idx]);
        ov::Layout layout = convert_layout_type (info.layout[idx]);

        output_info.tensor().set_element_type (precision);
        output_info.tensor().set_layout (layout);
        output_info.model().set_layout (layout);
        idx++;
    }

    _network = ppp.build();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnObjectDetection::get_bounding_boxes (const float* result_ptr,
                                        const uint32_t idx,
                                        std::vector<Vec4i> &boxes,
                                        std::vector<int32_t> &classes)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (!result_ptr) {
        XCAM_LOG_ERROR ("Inference results error!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    DnnInferInputOutputInfo output_infos;
    get_model_output_info (output_infos);

    uint32_t image_width = get_input_image_width (idx);
    uint32_t image_height = get_input_image_height (idx);
    uint32_t max_proposal_count = output_infos.channels[idx];
    uint32_t object_size = output_infos.object_size[idx];

    uint32_t box_count = 0;
    for (uint32_t cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
        float image_id = result_ptr[cur_proposal * object_size + 0];
        if (image_id < 0) {
            break;
        }

        float label = result_ptr[cur_proposal * object_size + 1];
        float confidence = result_ptr[cur_proposal * object_size + 2];
        float xmin = result_ptr[cur_proposal * object_size + 3] * image_width;
        float ymin = result_ptr[cur_proposal * object_size + 4] * image_height;
        float xmax = result_ptr[cur_proposal * object_size + 5] * image_width;
        float ymax = result_ptr[cur_proposal * object_size + 6] * image_height;

        if (confidence > 0.5) {
            classes.push_back(static_cast<int32_t>(label));
            boxes.push_back (Vec4i ( static_cast<int32_t>(xmin),
                                     static_cast<int32_t>(ymin),
                                     static_cast<int32_t>(xmax - xmin),
                                     static_cast<int32_t>(ymax - ymin) ));

            XCAM_LOG_DEBUG ("Proposal:%d label:%d confidence:%f", cur_proposal, classes[box_count], confidence);
            XCAM_LOG_DEBUG ("Boxes[%d] {%d, %d, %d, %d}",
                            box_count, boxes[box_count][0], boxes[box_count][1],
                            boxes[box_count][2], boxes[box_count][3]);
            box_count++;
        }
    }
    return XCAM_RETURN_NO_ERROR;
}

}  // namespace XCam
