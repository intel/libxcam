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
    set_output_layer_type ("Convolution");
}


DnnSuperResolution::~DnnSuperResolution ()
{

}

XCamReturn
DnnSuperResolution::set_output_layer_type (const char* type)
{
    _output_layer_type.insert (DnnOutputLayerType::value_type (DnnInferSuperResolution, type));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSuperResolution::get_model_input_info (DnnInferInputOutputInfo& info)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    InputsDataMap inputs_info (_network.getInputsInfo ());
    InputInfo::Ptr input_info = nullptr;

    int id = 0;
    InferenceEngine::Precision precision;
    for (auto & item : inputs_info) {
        if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
            input_info = item.second;
            XCAM_LOG_DEBUG ("Batch size is: %d", _network.getBatchSize());
            precision = Precision::U8;
            item.second->setPrecision(Precision::U8);
        } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
            precision = Precision::FP32;
            item.second->setPrecision(Precision::FP32);
            if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6)) {
                XCAM_LOG_ERROR ("Invalid input info. Should be 3 or 6 values length");
                return XCAM_RETURN_ERROR_PARAM;
            }
        }

        info.width[id] = inputs_info[item.first]->getTensorDesc().getDims()[3];
        info.height[id] = inputs_info[item.first]->getTensorDesc().getDims()[2];
        info.channels[id] = inputs_info[item.first]->getTensorDesc().getDims()[1];
        info.object_size[id] = inputs_info[item.first]->getTensorDesc().getDims()[0];
        info.precision[id] = convert_precision_type (precision);
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

    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
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
        idx ++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSuperResolution::get_model_output_info (DnnInferInputOutputInfo& info)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
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
            const InferenceEngine::SizeVector output_dims = output_info->getTensorDesc().getDims();

            info.object_size[idx] = output_dims[0];
            info.channels[idx] = output_dims[1];
            info.height[idx] = output_dims[2];
            info.width[idx] = output_dims[3];
            info.precision[idx] = convert_precision_type (output_info->getPrecision());
            info.layout[idx] = convert_layout_type (output_info->getLayout());
            info.data_type[idx] = DnnInferDataTypeImage;
            info.format[idx] = DnnInferImageFormatBGRPlanar;
            info.batch_size = idx + 1;
            info.numbers = outputs_info.size ();
        } else {
            XCAM_LOG_ERROR ("output data pointer is not valid");
            return XCAM_RETURN_ERROR_UNKNOWN;
        }
        idx ++;
        out.second->setPrecision (Precision::FP32);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSuperResolution::set_model_output_info (DnnInferInputOutputInfo& info)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
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

}  // namespace XCam
