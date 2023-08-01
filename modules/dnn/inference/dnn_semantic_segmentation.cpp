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
 * Author: Ali Mansouri <ali.m.t1992@gmail.com>
 */

#include <openvino/openvino.hpp>

#include "dnn_semantic_segmentation.h"
#include "dnn_inference_utils.h"

using namespace std;
using namespace ov;

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
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    for (size_t id = 0; id < get_input_size (); id ++) {
        const ov::PartialShape input_dims = _network->input (id).get_partial_shape ();
        if (input_dims.size() == 4) {
            XCAM_LOG_DEBUG ("Batch size is: %d", XCamDNN::convert_dim(input_dims[0]));
            info.width[id] = XCamDNN::convert_dim(input_dims[3]);
            info.height[id] = XCamDNN::convert_dim(input_dims[2]);
            info.channels[id] = XCamDNN::convert_dim(input_dims[1]);
            info.object_size[id] = XCamDNN::convert_dim(input_dims[0]);
            info.data_type[id] = DnnInferDataTypeImage;
            info.precision[id] = DnnInferPrecisionU8;
            info.layout[id] = DnnInferLayoutBCHW;
        } else if (input_dims.size() == 2) {
            info.precision[id] = DnnInferPrecisionFP32;
            if ((XCamDNN::convert_dim(input_dims[1]) != 3 && XCamDNN::convert_dim(input_dims[1]) != 6)) {
                XCAM_LOG_ERROR ("Invalid input info. Should be 3 or 6 values length");
                return XCAM_RETURN_ERROR_PARAM;
            }
        }
    }

    info.batch_size = XCamDNN::convert_dim(_network->input (0).get_partial_shape ()[0]);
    info.numbers = get_input_size ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSemanticSegmentation::set_model_input_info (DnnInferInputOutputInfo& info)
{
    XCAM_LOG_DEBUG ("DnnSemanticSegmentation::set_model_input_info");

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
DnnSemanticSegmentation::get_model_output_info (DnnInferInputOutputInfo& info)
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

        if (_network->output (idx).get_partial_shape ().size() == 4) {
            const ov::PartialShape output_dims = _network->output (idx).get_partial_shape ();

            info.object_size[idx] = XCamDNN::convert_dim(output_dims[0]);
            info.channels[idx] = XCamDNN::convert_dim(output_dims[1]);
            info.height[idx] = XCamDNN::convert_dim(output_dims[2]);
            info.width[idx] = XCamDNN::convert_dim(output_dims[3]);
            info.precision[idx] = DnnInferPrecisionFP32;
            info.layout[idx] = DnnInferLayoutBCHW;
            info.data_type[idx] = DnnInferDataTypeNonImage;
            info.format[idx] = DnnInferImageFormatGeneric1D;
        } else if (_network->output (idx).get_partial_shape ().size() == 3) {
            const ov::PartialShape output_dims = _network->output (idx).get_partial_shape ();

            info.object_size[idx] = 1;
            info.channels[idx] = 1;
            info.height[idx] = XCamDNN::convert_dim(output_dims[1]);
            info.width[idx] = XCamDNN::convert_dim(output_dims[2]);
            info.precision[idx] = DnnInferPrecisionFP32;
            info.layout[idx] = DnnInferLayoutBHW;
            info.data_type[idx] = DnnInferDataTypeNonImage;
            info.format[idx] = DnnInferImageFormatGeneric1D;
        } else if (_network->output (idx).get_partial_shape ().size() == 2) {
            const ov::PartialShape output_dims = _network->output (idx).get_partial_shape ();

            info.object_size[idx] = XCamDNN::convert_dim(output_dims[0]);
            info.channels[idx] = XCamDNN::convert_dim(output_dims[1]);
            info.height[idx] = 1;
            info.width[idx] = 1;
            info.precision[idx] = DnnInferPrecisionFP32;
            info.layout[idx] = DnnInferLayoutNC;
            info.data_type[idx] = DnnInferDataTypeNonImage;
            info.format[idx] = DnnInferImageFormatGeneric1D;
        } else if (_network->output (idx).get_partial_shape ().size() == 1) {
            const ov::PartialShape output_dims = _network->output (idx).get_partial_shape ();

            info.object_size[idx] = XCamDNN::convert_dim(output_dims[0]);
            info.channels[idx] = 1;
            info.height[idx] = 1;
            info.width[idx] = 1;
            info.precision[idx] = DnnInferPrecisionFP32;
            info.layout[idx] = DnnInferLayoutN;
            info.data_type[idx] = DnnInferDataTypeNonImage;
            info.format[idx] = DnnInferImageFormatGeneric1D;
        } else {
            XCAM_LOG_ERROR ("Dimension of output  %d is invalid!", idx);
            return XCAM_RETURN_ERROR_ORDER;
        }
    }

    info.batch_size = XCamDNN::convert_dim(_network->output (0).get_partial_shape ()[0]);
    info.numbers = get_output_size ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnSemanticSegmentation::set_model_output_info (DnnInferInputOutputInfo& info)
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
DnnSemanticSegmentation::get_segmentation_map (const std::vector<float*> result_ptr,
        const uint32_t idx,
        std::vector<std::vector<uint32_t>>& out_classes)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (result_ptr.empty()) {
        XCAM_LOG_ERROR ("Inference results error!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (get_output_size () == 1) {
        DnnInferInputOutputInfo output_infos;
        get_model_output_info (output_infos);

        uint32_t map_width = output_infos.width[0];
        uint32_t map_height = output_infos.height[0];
        uint32_t channels = output_infos.channels[0];
        uint32_t stride = map_width * map_height * channels;

        const auto output_data = result_ptr[0];

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
    } else if (get_output_size () == 3) {
        DnnInferInputOutputInfo output_infos;
        get_model_output_info (output_infos);
        DnnInferInputOutputInfo input_infos;
        get_model_output_info (input_infos);

        uint32_t map_width = input_infos.width[0];
        uint32_t map_height = input_infos.height[0];
        uint32_t channels = output_infos.channels[1];
        uint32_t max_proposal_count = _infer_request.get_output_tensor(0).get_shape ()[0];
        uint32_t stride0 = max_proposal_count;
        uint32_t stride1 = max_proposal_count * channels;
        uint32_t stride2 = max_proposal_count * output_infos.width[2] * output_infos.height[2];

        std::vector<std::vector<float>> out_prob (map_height, std::vector<float>(map_width, 0.0));

        for (uint32_t w = 0; w < map_width; w++) {
            for (uint32_t h = 0; h < map_height; h++) {
                out_classes[h][w] = 0;
                out_prob[h][w] = 1.0f;
                }
        }

        for (uint32_t cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
            float label = 0, confidence = 0, xmin = 0, ymin = 0, xmax = 0, ymax = 0;

            label = result_ptr[1][idx * stride0 + cur_proposal * channels + 0];
            confidence = result_ptr[0][idx * stride1 + cur_proposal * channels + 4];
            xmin = result_ptr[0][idx * stride1 + cur_proposal * channels + 0] * map_width;
            ymin = result_ptr[0][idx * stride1 + cur_proposal * channels + 1] * map_height;
            xmax = result_ptr[0][idx * stride1 + cur_proposal * channels + 2] * map_width;
            ymax = result_ptr[0][idx * stride1 + cur_proposal * channels + 3] * map_height;

            if (confidence > 0.5) {
                for (uint32_t w = xmin; w < xmax; w++) {
                    for (uint32_t h = ymin; h < ymax; h++) {
                        out_prob[h][w] = result_ptr[2][idx * stride2 + cur_proposal *  output_infos.height[2] * output_infos.width[2] +
                        static_cast<int>(((h - ymin) * output_infos.height[2]) / (ymax - ymin)) * output_infos.width[2] +
                        static_cast<int>(((w - xmin) * output_infos.width[2]) / (xmax - xmin))];
                        if (out_prob[h][w] > 0.5) {
                            out_classes[h][w] = label;
                        }
                    }
                }
            }
        }

    } else {
        XCAM_LOG_ERROR ("Number of outputs is invalid!");
        return XCAM_RETURN_ERROR_ORDER;
    }
    return XCAM_RETURN_NO_ERROR;
}

}  // namespace XCam
