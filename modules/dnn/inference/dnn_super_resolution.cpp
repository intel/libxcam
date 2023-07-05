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
 * Author: Ali Mansouri <ali.m.t1992@gmail.com>
 */

#include <openvino/openvino.hpp>

#include "dnn_super_resolution.h"

using namespace std;
using namespace ov;

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
DnnSuperResolution::set_model_input_info (DnnInferInputOutputInfo& info)
{
    XCAM_LOG_DEBUG ("DnnSuperResolution::set_model_input_info");

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
DnnSuperResolution::get_model_output_info (DnnInferInputOutputInfo& info)
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

        info.object_size[idx] = output_dims[0];
        info.channels[idx] = output_dims[1];
        info.height[idx] = output_dims[2];
        info.width[idx] = output_dims[3];
        info.precision[idx] = DnnInferPrecisionFP32;
        info.layout[idx] = DnnInferLayoutNCHW;
        info.data_type[idx] = DnnInferDataTypeImage;
        info.format[idx] = DnnInferImageFormatBGRPlanar;
    }

    info.batch_size = _network->output (0).get_shape ()[0];
    info.numbers = get_output_size ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSuperResolution::set_model_output_info (DnnInferInputOutputInfo& info)
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

}  // namespace XCam
