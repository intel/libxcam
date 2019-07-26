/*
 * dnn_semantic_segmentation.cpp -  semantic segmentation
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

#include <inference_engine.hpp>

#include "dnn_semantic_segmentation.h"

using namespace std;
using namespace InferenceEngine;

namespace XCam {

DnnSemanticSegmentation::DnnSemanticSegmentation (DnnInferConfig& config)
    : DnnInferenceEngine (config)
{
    XCAM_LOG_DEBUG ("DnnSemanticSegmentation::DnnSemanticSegmentation");
    set_output_layer_type ("ArgMax");
}

DnnSemanticSegmentation::~DnnSemanticSegmentation ()
{

}

XCamReturn
DnnSemanticSegmentation::set_output_layer_type (const char* type)
{
    _output_layer_type.insert (DnnOutputLayerType::value_type (DnnInferSemanticSegmentation, type));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSemanticSegmentation::get_model_input_info (DnnInferInputOutputInfo& info)
{
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    int id = 0;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    for (auto & in : inputs_info) {
        auto& input = in.second;
        const InferenceEngine::SizeVector input_dims = input->getDims ();

        info.width[id] = input_dims[0];
        info.height[id] = input_dims[1];
        info.channels[id] = input_dims[2];
        info.object_size[id] = input_dims[3];
        info.precision[id] = convert_precision_type (input->getPrecision());
        info.layout[id] = convert_layout_type (input->getLayout());

        in.second->setPrecision(Precision::U8);

        id++;
    }
    info.batch_size = get_batch_size ();
    info.numbers = inputs_info.size ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSemanticSegmentation::set_model_input_info (DnnInferInputOutputInfo& info)
{
    XCAM_LOG_DEBUG ("DnnSemanticSegmentation::set_model_input_info");

    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    InputsDataMap inputs_info (_network.getInputsInfo ());
    if (info.numbers != inputs_info.size ()) {
        XCAM_LOG_ERROR ("Input size is not matched with model info numbers %d !", info.numbers);
        return XCAM_RETURN_ERROR_PARAM;
    }

    int idx = 0;
    for (auto & in : inputs_info) {
        Precision precision = convert_precision_type (info.precision[idx]);
        in.second->setPrecision (precision);
        Layout layout = convert_layout_type (info.layout[idx]);
        in.second->setLayout (layout);
        idx++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSemanticSegmentation::get_model_output_info (DnnInferInputOutputInfo& info)
{
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    std::string output_name;
    OutputsDataMap outputs_info (_network.getOutputsInfo ());
    DataPtr output_info;
    uint32_t idx = 0;

    for (const auto& out : outputs_info) {
        if (output_name.empty ()) {
            output_name = out.first;
        }

        output_info = out.second;
        if (output_info.get ()) {
            const InferenceEngine::SizeVector output_dims = output_info->getTensorDesc ().getDims ();

            info.object_size[idx] = output_dims[0];
            info.channels[idx] = output_dims[1];
            info.height[idx] = output_dims[2];
            info.width[idx] = output_dims[3];
            info.precision[idx] = convert_precision_type (output_info->getPrecision ());
            info.layout[idx] = convert_layout_type (output_info->getLayout ());
            info.data_type[idx] = DnnInferDataTypeNonImage;
            info.format[idx] = DnnInferImageFormatGeneric1D;
            info.batch_size = idx + 1;
            info.numbers = outputs_info.size ();
        } else {
            XCAM_LOG_ERROR ("output data pointer is not valid");
            return XCAM_RETURN_ERROR_UNKNOWN;
        }
        idx ++;
        out.second->setPrecision (Precision::FP32);
    }


    if (output_info.get ()) {
    } else {
        XCAM_LOG_ERROR ("Get output info error!");
        return XCAM_RETURN_ERROR_UNKNOWN;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSemanticSegmentation::set_model_output_info (DnnInferInputOutputInfo& info)
{
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo ());
    if (info.numbers != outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output size is not matched with model!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    int idx = 0;
    for (auto & out : outputs_info) {
        Precision precision = convert_precision_type (info.precision[idx]);
        out.second->setPrecision (precision);
        Layout layout = convert_layout_type (info.layout[idx]);
        out.second->setLayout (layout);
        idx++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSemanticSegmentation::get_segmentation_map (const float* result_ptr,
        const uint32_t idx,
        std::vector<std::vector<uint32_t>>& out_classes)
{
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (!result_ptr) {
        XCAM_LOG_ERROR ("Inference results error!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    DnnInferInputOutputInfo output_infos;
    get_model_output_info (output_infos);

    uint32_t map_width = output_infos.width[idx];
    uint32_t map_height = output_infos.height[idx];
    uint32_t channels = output_infos.channels[idx];
    //uint32_t object_size = output_infos.object_size[idx];
    uint32_t stride = map_width * map_height * channels;

    const auto output_data = result_ptr;

    std::vector<std::vector<float>> out_prob (map_height, std::vector<float>(map_width, 0.0));

    for (uint32_t w = 0; w < map_width; w++) {
        for (uint32_t h = 0; h < map_height; h++) {
            if (channels == 1) {
                out_classes[h][w] = output_data[stride * idx + map_width * h + w];
            } else {
                for (uint32_t ch = 0; ch < channels; ch++) {
                    auto data = output_data[stride * idx + map_width * map_height * ch + map_width * h + w];
                    if (data > out_prob[h][w]) {
                        out_classes[h][w] = ch;
                        out_prob[h][w] = data;
                    }
                }
            }
        }
    }
    return XCAM_RETURN_NO_ERROR;
}

}  // namespace XCam
