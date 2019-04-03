/*
 * dnn_super_resolution.cpp -  super resolution
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

#include "dnn_super_resolution.h"

using namespace std;
using namespace InferenceEngine;

namespace XCam {

DnnSuperResolution::DnnSuperResolution (DnnInferConfig& config)
    : DnnInferenceEngine (config)
{
    XCAM_LOG_DEBUG ("DnnSuperResolution::DnnSuperResolution");
}


DnnSuperResolution::~DnnSuperResolution ()
{

}

XCamReturn
DnnSuperResolution::get_model_input_info (DnnInferInputOutputInfo& info)
{
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    int id = 0;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    for (auto & item : inputs_info) {
        auto& input = item.second;
        const InferenceEngine::SizeVector input_dims = input->getDims ();

        info.width[id] = input_dims[0];
        info.height[id] = input_dims[1];
        info.channels[id] = input_dims[2];
        info.object_size[id] = input_dims[3];
        info.precision[id] = convert_precision_type (input->getPrecision());
        info.layout[id] = convert_layout_type (input->getLayout());

        item.second->setPrecision(Precision::U8);

        id++;
    }
    info.batch_size = get_batch_size ();
    info.numbers = inputs_info.size ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSuperResolution::set_model_input_info (DnnInferInputOutputInfo& info)
{
    XCAM_LOG_DEBUG ("DnnSuperResolution::set_model_input_info");

    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    InputsDataMap inputs_info (_network.getInputsInfo ());
    if (info.numbers != inputs_info.size ()) {
        XCAM_LOG_ERROR ("Input size is not matched with model info numbers %d !", info.numbers);
        return XCAM_RETURN_ERROR_PARAM;
    }
    int id = 0;

    for (auto & item : inputs_info) {
        Precision precision = convert_precision_type (info.precision[id]);
        item.second->setPrecision (precision);
        Layout layout = convert_layout_type (info.layout[id]);
        item.second->setLayout (layout);
        id++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSuperResolution::get_model_output_info (DnnInferInputOutputInfo& info)
{
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    std::string output_name;
    OutputsDataMap outputs_info (_network.getOutputsInfo ());
    DataPtr output_info;
    for (const auto& out : outputs_info) {
        if (output_name.empty ()) {
            output_name = out.first;
        }

        output_info = out.second;
        if (!output_info) {
            XCAM_LOG_ERROR ("output data pointer is not valid");
            return XCAM_RETURN_ERROR_UNKNOWN;
        }

        out.second->setPrecision (Precision::FP32);
    }

    uint32_t id = 0;

    if (output_info.get ()) {
        const InferenceEngine::SizeVector output_dims = output_info->getTensorDesc().getDims();

        info.object_size[id] = output_dims[0];
        info.channels[id] = output_dims[1];
        info.width[id]    = output_dims[2];
        info.height[id]   = output_dims[3];

        info.precision[id] = convert_precision_type (output_info->getPrecision());
        info.layout[id] = convert_layout_type (output_info->getLayout());

        info.batch_size = 1;
        info.numbers = outputs_info.size ();
    } else {
        XCAM_LOG_ERROR ("Get output info error!");
        return XCAM_RETURN_ERROR_UNKNOWN;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSuperResolution::set_model_output_info (DnnInferInputOutputInfo& info)
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

    int id = 0;
    for (auto & item : outputs_info) {
        Precision precision = convert_precision_type (info.precision[id]);
        item.second->setPrecision (precision);
        Layout layout = convert_layout_type (info.layout[id]);
        item.second->setLayout (layout);
        id++;
    }

    return XCAM_RETURN_NO_ERROR;
}

void*
DnnSuperResolution::get_inference_results (uint32_t idx, uint32_t& size)
{
    if (! _model_created || ! _model_loaded) {
        XCAM_LOG_ERROR ("Please create and load the model firstly!");
        return NULL;
    }
    std::string item_name;

    OutputsDataMap outputs_info (_network.getOutputsInfo ());
    if (idx > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return NULL;
    }

    for (auto & item : outputs_info) {
        if (item.second->creatorLayer.lock()->type == "Convolution") {
            item_name = item.first;
            break;
        }
    }

    if (item_name.empty ()) {
        XCAM_LOG_ERROR ("item name is empty!");
        return NULL;
    }

    const Blob::Ptr output_blob = _infer_request.GetBlob (item_name);
    const auto output_data = output_blob->buffer ().as<PrecisionTrait<Precision::FP32>::value_type*> ();

    size = output_blob->byteSize ();

    return (reinterpret_cast<void *>(output_data));
}

}  // namespace XCam
