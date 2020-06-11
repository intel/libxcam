/*
 * dnn_inference_engine.cpp -  dnn inference engine
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

#include "dnn_inference_engine.h"
#include "dnn_inference_utils.h"

#include <iomanip>
#include <format_reader_ptr.h>
#include <ngraph/ngraph.hpp>

#if HAVE_OPENCV
#include "ocv/cv_std.h"
#endif

using namespace std;
using namespace InferenceEngine;

namespace XCam {

DnnInferenceEngine::DnnInferenceEngine (DnnInferConfig& config)
    : _model_loaded (false)
    , _model_type (config.model_type)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::DnnInferenceEngine");
    _input_image_width.clear ();
    _input_image_height.clear ();

    create_model (config);
}

DnnInferenceEngine::~DnnInferenceEngine ()
{

}

std::vector<std::string>
DnnInferenceEngine::get_available_devices ()
{
    std::vector<std::string> dev;
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_INFO ("Please create inference engine");
        return dev;
    }
    return _ie->GetAvailableDevices ();
}

XCamReturn
DnnInferenceEngine::create_model (DnnInferConfig& config)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::create_model");
    if (_ie.ptr ()) {
        XCAM_LOG_INFO ("model already created!");
        return XCAM_RETURN_NO_ERROR;
    }

    XCAM_LOG_DEBUG ("pre-trained model file name: %s", config.model_filename.c_str ());
    if ("" == config.model_filename) {
        XCAM_LOG_ERROR ("Model file name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }
    _ie = new InferenceEngine::Core ();

    if ( (DnnInferDeviceCPU == config.target_id) && ("" != config.mkldnn_ext)) {
        XCAM_LOG_DEBUG ("Load CPU MKLDNN extensions %s", config.mkldnn_ext.c_str ());
        auto extensionPtr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension> (config.mkldnn_ext);
        _ie->AddExtension (extensionPtr);
    } else if ((DnnInferDeviceGPU == config.target_id) && ("" != config.cldnn_ext)) {
        XCAM_LOG_DEBUG ("Load GPU extensions: %s", config.cldnn_ext.c_str ());
        _ie->SetConfig ({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, config.cldnn_ext } }, "GPU");
    }

    _network = _ie->ReadNetwork (get_filename_prefix (config.model_filename) + ".xml");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::load_model (DnnInferConfig& config)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::load_model, device name:%s", config.device_name.c_str ());
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }
    if (_model_loaded) {
        XCAM_LOG_INFO ("model already loaded!");
        return XCAM_RETURN_NO_ERROR;
    }

    InferenceEngine::ExecutableNetwork execute_network = _ie->LoadNetwork (_network, config.device_name, config.config_file);

    _infer_request = execute_network.CreateInferRequest ();

    _model_loaded = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::get_info (DnnInferenceEngineInfo& info)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    info.major = GetInferenceEngineVersion ()->apiVersion.major;
    info.minor = GetInferenceEngineVersion ()->apiVersion.minor;
    info.desc = GetInferenceEngineVersion ()->description;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_batch_size (const size_t size)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    _network.setBatchSize (size);
    return XCAM_RETURN_NO_ERROR;
}

size_t
DnnInferenceEngine::get_batch_size ()
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return -1;
    }

    return _network.getBatchSize ();
}

XCamReturn
DnnInferenceEngine::start (bool sync)
{
    XCAM_LOG_DEBUG ("Start inference %s", sync ? "Sync" : "Async");

    if (! _model_loaded) {
        XCAM_LOG_ERROR ("Please load the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (sync) {
        _infer_request.Infer ();
    } else {
        _infer_request.StartAsync ();
        _infer_request.Wait (IInferRequest::WaitMode::RESULT_READY);
    }

    return XCAM_RETURN_NO_ERROR;
}

size_t
DnnInferenceEngine::get_input_size ()
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return -1;
    }

    InputsDataMap inputs_info (_network.getInputsInfo());
    return inputs_info.size ();
}

size_t
DnnInferenceEngine::get_output_size ()
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return -1;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo());
    return outputs_info.size ();
}

XCamReturn
DnnInferenceEngine::set_input_precision (uint32_t idx, DnnInferPrecisionType precision)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    InputsDataMap inputs_info (_network.getInputsInfo ());

    if (idx > inputs_info.size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    uint32_t i = 0;
    for (auto & in : inputs_info) {
        if (i == idx) {
            Precision input_precision = convert_precision_type (precision);
            in.second->setPrecision (input_precision);
            break;
        }
        i++;
    }

    return XCAM_RETURN_NO_ERROR;
}

DnnInferPrecisionType
DnnInferenceEngine::get_input_precision (uint32_t idx)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return DnnInferPrecisionUnspecified;
    }

    DnnInferInputOutputInfo inputs_info;
    get_model_input_info (inputs_info);

    if (idx > get_input_size ()) {
        XCAM_LOG_ERROR ("Index is out of range");
        return DnnInferPrecisionUnspecified;
    }

    return inputs_info.precision[idx];
}

XCamReturn
DnnInferenceEngine::set_output_precision (uint32_t idx, DnnInferPrecisionType precision)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo ());

    if (idx > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    uint32_t i = 0;
    for (auto & out : outputs_info) {
        if (i == idx) {
            Precision output_precision = convert_precision_type (precision);
            out.second->setPrecision (output_precision);
            break;
        }
        i++;
    }

    return XCAM_RETURN_NO_ERROR;
}

DnnInferPrecisionType
DnnInferenceEngine::get_output_precision (uint32_t idx)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return DnnInferPrecisionUnspecified;
    }

    DnnInferInputOutputInfo outputs_info;
    get_model_output_info (outputs_info);

    if (idx > get_output_size ()) {
        XCAM_LOG_ERROR ("Index is out of range");
        return DnnInferPrecisionUnspecified;
    }

    return outputs_info.precision[idx];
}

DnnInferImageFormatType
DnnInferenceEngine::get_output_format (uint32_t idx)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return DnnInferImageFormatUnknown;
    }

    DnnInferInputOutputInfo outputs_info;
    get_model_output_info (outputs_info);

    if (idx > get_output_size ()) {
        XCAM_LOG_ERROR ("Index is out of range");
        return DnnInferImageFormatUnknown;
    }

    return outputs_info.format[idx];
}

XCamReturn
DnnInferenceEngine::set_input_layout (uint32_t idx, DnnInferLayoutType layout)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }
    InputsDataMap inputs_info (_network.getInputsInfo ());

    if (idx > inputs_info.size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    uint32_t i = 0;
    for (auto & in : inputs_info) {
        if (i == idx) {
            Layout input_layout = convert_layout_type (layout);
            in.second->setLayout (input_layout);
            break;
        }
        i++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_output_layout (uint32_t idx, DnnInferLayoutType layout)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo ());

    if (idx > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    uint32_t i = 0;
    for (auto & out : outputs_info) {
        if (i == idx) {
            Layout output_layout = convert_layout_type (layout);
            out.second->setLayout (output_layout);
            break;
        }
        i++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_input_blob (uint32_t idx, DnnInferData& data)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    unsigned int id = 0;
    std::string input_name;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    if (idx > inputs_info.size()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (auto & in : inputs_info) {
        if (id == idx) {
            input_name = in.first;
            break;
        }
        id++;
    }

    if (input_name.empty ()) {
        XCAM_LOG_ERROR ("input name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (data.batch_idx > get_batch_size ()) {
        XCAM_LOG_ERROR ("Too many input, it is bigger than batch size!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    Blob::Ptr blob = _infer_request.GetBlob (input_name);
    if (data.precision == DnnInferPrecisionFP32) {
        if (data.data_type == DnnInferDataTypeImage) {
            copy_image_to_blob<PrecisionTrait<Precision::FP32>::value_type>(data, blob, data.batch_idx);
        } else {
            copy_data_to_blob<PrecisionTrait<Precision::FP32>::value_type>(data, blob, data.batch_idx);
        }
    } else {
        if (data.data_type == DnnInferDataTypeImage) {
            copy_image_to_blob<uint8_t>(data, blob, data.batch_idx);
        } else {
            copy_data_to_blob<uint8_t>(data, blob, data.batch_idx);
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_inference_data (std::vector<std::string> images)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    uint32_t idx = 0;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    for (auto & i : images) {
        FormatReader::ReaderPtr reader (i.c_str ());
        if (reader.get () == NULL) {
            XCAM_LOG_WARNING ("Image %d cannot be read!", i);
            continue;
        }

        _input_image_width.push_back (reader->width ());
        _input_image_height.push_back (reader->height ());

        uint32_t image_width = 0;
        uint32_t image_height = 0;

        for (auto & in : inputs_info) {
            image_width = inputs_info[in.first]->getTensorDesc().getDims()[3];
            image_height = inputs_info[in.first]->getTensorDesc().getDims()[2];
        }

        std::shared_ptr<unsigned char> data (reader->getData (image_width, image_height));

        if (data.get () != NULL) {
            DnnInferData image;
            image.width = image_width;
            image.height = image_height;
            image.width_stride = image_width;
            image.height_stride = image_height;
            image.buffer = data.get ();
            image.channel_num = 3;
            image.batch_idx = idx;
            image.image_format = DnnInferImageFormatBGRPacked;

            // set precision & data type
            image.precision = get_input_precision (idx);
            image.data_type = DnnInferDataTypeImage;

            set_input_blob (idx, image);
            idx ++;
        } else {
            XCAM_LOG_WARNING ("Valid input images were not found!");
            continue;
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_inference_data (const VideoBufferList& images)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    uint32_t idx = 0;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    for (VideoBufferList::const_iterator iter = images.begin(); iter != images.end (); ++iter) {
        SmartPtr<VideoBuffer> buf = *iter;
        XCAM_ASSERT (buf.ptr ());

        VideoBufferInfo buf_info = buf->get_video_info ();
        _input_image_width.push_back (buf_info.width);
        _input_image_height.push_back (buf_info.height);

        uint32_t image_width = 0;
        uint32_t image_height = 0;

        for (auto & in : inputs_info) {
            image_width = inputs_info[in.first]->getTensorDesc().getDims()[3];
            image_height = inputs_info[in.first]->getTensorDesc().getDims()[2];
        }

        float x_ratio = float(image_width) / float(buf_info.width);
        float y_ratio = float(image_height) / float(buf_info.height);

        uint8_t* data = NULL;
        if (buf_info.format == V4L2_PIX_FMT_NV12) {
            data = XCamDNN::convert_NV12_to_BGR (buf, x_ratio, y_ratio);
        } else if (buf_info.format == V4L2_PIX_FMT_BGR24) {
            data = buf->map ();
        }

        if (data != NULL) {
            DnnInferData image;
            image.width = image_width;
            image.height = image_height;
            image.width_stride = image_width;
            image.height_stride = image_height;
            image.buffer = data;
            image.channel_num = 3;
            image.batch_idx = idx;
            image.image_format = DnnInferImageFormatBGRPacked;

            // set precision & data type
            image.precision = get_input_precision (idx);
            image.data_type = DnnInferDataTypeImage;

            set_input_blob (idx, image);
            idx ++;
        } else {
            XCAM_LOG_WARNING ("Valid input images were not found!");
            continue;
        }

        if (buf_info.format != V4L2_PIX_FMT_NV12) {
            buf->unmap ();
        }
    }

    return XCAM_RETURN_NO_ERROR;
}


std::shared_ptr<uint8_t>
DnnInferenceEngine::read_input_image (std::string& image)
{
    FormatReader::ReaderPtr reader (image.c_str ());
    if (reader.get () == NULL) {
        XCAM_LOG_WARNING ("Image cannot be read!");
        return NULL;
    }

    uint32_t image_width = reader->width ();
    uint32_t image_height = reader->height ();

    std::shared_ptr<uint8_t> data (reader->getData (image_width, image_height));

    if (data.get () != NULL) {
        return data;
    } else {
        XCAM_LOG_WARNING ("Valid input images were not found!");
        return NULL;
    }
}

XCamReturn
DnnInferenceEngine::save_output_image (const std::string& image_name, uint32_t index)
{
    if (NULL == _ie.ptr () || ! _model_loaded) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (NULL == _output_layer_type[_model_type]) {
        XCAM_LOG_ERROR ("Please set model output layer type!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo ());
    if (index > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    std::string output_name;
    DataPtr output_info;
    if (auto ngraphFunction = _network.getFunction()) {
        for (const auto& out : outputs_info) {
            for (const auto & op : ngraphFunction->get_ops()) {
                if (op->get_type_info() == ngraph::op::DetectionOutput::type_info &&
                        op->get_friendly_name() == out.second->getName()) {
                    output_name = out.first;
                    output_info = out.second;
                    break;
                }
            }
        }
    } else {
        output_info = outputs_info.begin()->second;
        output_name = output_info->getName();
    }

    if (output_name.empty ()) {
        XCAM_LOG_ERROR ("out name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    const Blob::Ptr output_blob = _infer_request.GetBlob (output_name);
    const auto output_data = output_blob->buffer ().as<PrecisionTrait<Precision::FP32>::value_type*> ();

    size_t image_count = output_blob->getTensorDesc ().getDims ()[0];
    size_t channels = output_blob->getTensorDesc ().getDims ()[1];
    size_t image_height = output_blob->getTensorDesc ().getDims ()[2];
    size_t image_width = output_blob->getTensorDesc ().getDims ()[3];
    size_t pixel_count = image_width * image_height;

    XCAM_LOG_DEBUG ("Output size [image count, channels, width, height]: %d, %d, %d, %d",
                    image_count, channels, image_width, image_height);

    if (index > image_count) {
        return XCAM_RETURN_ERROR_PARAM;
    }

#if HAVE_OPENCV
    std::vector<cv::Mat> image_planes;
    if (3 == channels) {
        image_planes.push_back (cv::Mat (image_height, image_width, CV_32FC1, &(output_data[index * pixel_count * channels + pixel_count * 2])));
        image_planes.push_back (cv::Mat (image_height, image_width, CV_32FC1, &(output_data[index * pixel_count * channels + pixel_count])));
        image_planes.push_back (cv::Mat (image_height, image_width, CV_32FC1, &(output_data[index * pixel_count * channels])));
    } else if (1 == channels) {
        image_planes.push_back (cv::Mat (image_height, image_width, CV_32FC1, &(output_data[index * pixel_count])));
    }

    for (auto & image : image_planes) {
        image.convertTo (image, CV_8UC1, 255);
    }
    cv::Mat result_image;
    cv::merge (image_planes, result_image);
    cv::imwrite (image_name.c_str (), result_image);
#else
    if (3 == channels) {
        XCamDNN::save_bmp_file (image_name,
                                &output_data[index * pixel_count * channels],
                                get_output_format (index),
                                get_output_precision (index),
                                image_width,
                                image_height);
    }
#endif

    return XCAM_RETURN_NO_ERROR;
}

void*
DnnInferenceEngine::get_inference_results (uint32_t idx, uint32_t& size)
{
    if (NULL == _ie.ptr () || ! _model_loaded) {
        XCAM_LOG_ERROR ("Please create and load the model firstly!");
        return NULL;
    }

    std::string output_name;
    OutputsDataMap outputs_info (_network.getOutputsInfo ());
    if (idx > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return NULL;
    }

    if (NULL == _output_layer_type[_model_type]) {
        XCAM_LOG_ERROR ("Please set model output layer type!");
        return NULL;
    }

    DataPtr output_info;
    if (auto ngraphFunction = _network.getFunction()) {
        for (const auto& out : outputs_info) {
            for (const auto & op : ngraphFunction->get_ops()) {
                if (op->get_type_info() == ngraph::op::DetectionOutput::type_info &&
                        op->get_friendly_name() == out.second->getName()) {
                    output_name = out.first;
                    output_info = out.second;
                    break;
                }
            }
        }
    } else {
        output_info = outputs_info.begin()->second;
        output_name = output_info->getName();
    }

    if (output_name.empty ()) {
        XCAM_LOG_ERROR ("out name is empty!");
        return NULL;
    }

    const Blob::Ptr blob = _infer_request.GetBlob (output_name);
    float* output_result = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(blob->buffer ());

    size = blob->byteSize ();

    return (reinterpret_cast<void *>(output_result));
}

InferenceEngine::Layout
DnnInferenceEngine::estimate_layout_type (const int ch_num)
{
    if (ch_num == 4) {
        return InferenceEngine::Layout::NCHW;
    } else if (ch_num == 3) {
        return InferenceEngine::Layout::CHW;
    } else if (ch_num == 2) {
        return InferenceEngine::Layout::NC;
    } else {
        return InferenceEngine::Layout::ANY;
    }
}

InferenceEngine::Layout
DnnInferenceEngine::convert_layout_type (DnnInferLayoutType layout)
{
    switch (layout) {
    case DnnInferLayoutNCHW:
        return InferenceEngine::Layout::NCHW;
    case DnnInferLayoutNHWC:
        return InferenceEngine::Layout::NHWC;
    case DnnInferLayoutOIHW:
        return InferenceEngine::Layout::OIHW;
    case DnnInferLayoutC:
        return InferenceEngine::Layout::C;
    case DnnInferLayoutCHW:
        return InferenceEngine::Layout::CHW;
    case DnnInferLayoutHW:
        return InferenceEngine::Layout::HW;
    case DnnInferLayoutNC:
        return InferenceEngine::Layout::NC;
    case DnnInferLayoutCN:
        return InferenceEngine::Layout::CN;
    case DnnInferLayoutBlocked:
        return InferenceEngine::Layout::BLOCKED;
    case DnnInferLayoutAny:
        return InferenceEngine::Layout::ANY;
    default:
        return InferenceEngine::Layout::ANY;
    }
}

DnnInferLayoutType
DnnInferenceEngine::convert_layout_type (InferenceEngine::Layout layout)
{
    switch (layout) {
    case InferenceEngine::Layout::NCHW:
        return DnnInferLayoutNCHW;
    case InferenceEngine::Layout::NHWC:
        return DnnInferLayoutNHWC;
    case InferenceEngine::Layout::OIHW:
        return DnnInferLayoutOIHW;
    case InferenceEngine::Layout::C:
        return DnnInferLayoutC;
    case InferenceEngine::Layout::CHW:
        return DnnInferLayoutCHW;
    case InferenceEngine::Layout::HW:
        return DnnInferLayoutHW;
    case InferenceEngine::Layout::NC:
        return DnnInferLayoutNC;
    case InferenceEngine::Layout::CN:
        return DnnInferLayoutCN;
    case InferenceEngine::Layout::BLOCKED:
        return DnnInferLayoutBlocked;
    case InferenceEngine::Layout::ANY:
        return DnnInferLayoutAny;
    default:
        return DnnInferLayoutAny;
    }
}

InferenceEngine::Precision
DnnInferenceEngine::convert_precision_type (DnnInferPrecisionType precision)
{
    switch (precision) {
    case DnnInferPrecisionU8:
        return InferenceEngine::Precision::U8;
    case DnnInferPrecisionI8:
        return InferenceEngine::Precision::I8;
    case DnnInferPrecisionU16:
        return InferenceEngine::Precision::U16;
    case DnnInferPrecisionI16:
        return InferenceEngine::Precision::I16;
    case DnnInferPrecisionQ78:
        return InferenceEngine::Precision::Q78;
    case DnnInferPrecisionFP16:
        return InferenceEngine::Precision::FP16;
    case DnnInferPrecisionI32:
        return InferenceEngine::Precision::I32;
    case DnnInferPrecisionFP32:
        return InferenceEngine::Precision::FP32;
    case DnnInferPrecisionMixed:
        return InferenceEngine::Precision::MIXED;
    case DnnInferPrecisionCustom:
        return InferenceEngine::Precision::CUSTOM;
    case DnnInferPrecisionUnspecified:
        return InferenceEngine::Precision::UNSPECIFIED;
    default:
        return InferenceEngine::Precision::UNSPECIFIED;
    }
}

DnnInferPrecisionType
DnnInferenceEngine::convert_precision_type (InferenceEngine::Precision precision)
{
    switch (precision) {
    case InferenceEngine::Precision::MIXED:
        return DnnInferPrecisionMixed;
    case InferenceEngine::Precision::FP32:
        return DnnInferPrecisionFP32;
    case InferenceEngine::Precision::FP16:
        return DnnInferPrecisionFP16;
    case InferenceEngine::Precision::Q78:
        return DnnInferPrecisionQ78;
    case InferenceEngine::Precision::I16:
        return DnnInferPrecisionI16;
    case InferenceEngine::Precision::U8:
        return DnnInferPrecisionU8;
    case InferenceEngine::Precision::I8:
        return DnnInferPrecisionI8;
    case InferenceEngine::Precision::U16:
        return DnnInferPrecisionU16;
    case InferenceEngine::Precision::I32:
        return DnnInferPrecisionI32;
    case InferenceEngine::Precision::CUSTOM:
        return DnnInferPrecisionCustom;
    case InferenceEngine::Precision::UNSPECIFIED:
        return DnnInferPrecisionUnspecified;
    default:
        return DnnInferPrecisionUnspecified;
    }
}

std::string
DnnInferenceEngine::get_filename_prefix (const std::string &file_path)
{
    auto pos = file_path.rfind ('.');
    if (pos == std::string::npos) {
        return file_path;
    }

    return file_path.substr (0, pos);
}

template <typename T> XCamReturn
DnnInferenceEngine::copy_image_to_blob (const DnnInferData& data, Blob::Ptr& image_blob, int batch_index)
{
    // Filling input tensor with images. in order of b channel, then g and r channels
    MemoryBlob::Ptr image_ptr = as<MemoryBlob>(image_blob);
    if (!image_ptr) {
        XCAM_LOG_ERROR ("Can not cast imageInput to MemoryBlob");
        return XCAM_RETURN_ERROR_PARAM;
    }

    auto input_holder = image_ptr->wmap();

    size_t channels = image_ptr->getTensorDesc().getDims()[1];
    const size_t image_width = image_ptr->getTensorDesc().getDims()[3];
    const size_t image_height = image_ptr->getTensorDesc().getDims()[2];
    size_t image_size =  image_width * image_height;

    T* blob_data = input_holder.as<T*>();
    unsigned char* buffer = (unsigned char*)data.buffer;

    if (image_width != data.width || image_height != data.height) {
        XCAM_LOG_ERROR ("Input Image size (%dx%d) is not matched with model required size (%dx%d)!",
                        data.width, data.height, image_width, image_height);
        return XCAM_RETURN_ERROR_PARAM;
    }

    int batch_offset = batch_index * image_height * image_width * channels;

    if (DnnInferImageFormatBGRPlanar == data.image_format) {
        // B G R planar input image
        size_t image_stride_size = data.height_stride * data.width_stride;

        if (data.width == data.width_stride &&
                data.height == data.height_stride) {
            std::memcpy (blob_data + batch_offset, buffer, image_size * channels);
        } else if (data.width == data.width_stride) {
            for (size_t ch = 0; ch < channels; ++ch) {
                std::memcpy (blob_data + batch_offset + ch * image_size, buffer + ch * image_stride_size, image_size);
            }
        } else {
            for (size_t ch = 0; ch < channels; ch++) {
                for (size_t h = 0; h < image_height; h++) {
                    std::memcpy (blob_data + batch_offset + ch * image_size + h * image_width, buffer + ch * image_stride_size + h * data.width_stride, image_width);
                }
            }
        }
    } else if (DnnInferImageFormatBGRPacked == data.image_format) {
        for (size_t pid = 0; pid < image_size; pid++) {
            for (size_t ch = 0; ch < channels; ch++) {
                blob_data[batch_offset + ch * image_size + pid] = buffer[pid * channels + ch];
            }
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

template <typename T> XCamReturn
DnnInferenceEngine::copy_data_to_blob (const DnnInferData& data, Blob::Ptr& blob, int batch_index)
{
    T * buffer = (T *)data.buffer;
    T* blob_data = blob->buffer ().as<T*>();

    int batch_offset = batch_index * data.size;

    memcpy (blob_data + batch_offset, buffer, data.size);

    return XCAM_RETURN_NO_ERROR;
}

void
DnnInferenceEngine::print_performance_counts (const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performance_map)
{
    long long total_time = 0;
    XCAM_LOG_DEBUG ("performance counts:");
    for (const auto & it : performance_map) {
        std::string to_print(it.first);
        const int max_layer_name = 30;

        if (it.first.length () >= max_layer_name) {
            to_print = it.first.substr (0, max_layer_name - 4);
            to_print += "...";
        }

        XCAM_LOG_DEBUG ("layer: %s", to_print.c_str ());
        switch (it.second.status) {
        case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
            XCAM_LOG_DEBUG ("EXECUTED");
            break;
        case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
            XCAM_LOG_DEBUG ("NOT_RUN");
            break;
        case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            XCAM_LOG_DEBUG ("OPTIMIZED_OUT");
            break;
        }
        XCAM_LOG_DEBUG ("layerType: %s", std::string (it.second.layer_type).c_str ());
        XCAM_LOG_DEBUG ("realTime: %d", it.second.realTime_uSec);
        XCAM_LOG_DEBUG ("cpu: %d", it.second.cpu_uSec);
        XCAM_LOG_DEBUG ("execType: %s", it.second.exec_type);
        if (it.second.realTime_uSec > 0) {
            total_time += it.second.realTime_uSec;
        }
    }
    XCAM_LOG_DEBUG ("Total time: %d microseconds", total_time);
}

}  // namespace XCam
