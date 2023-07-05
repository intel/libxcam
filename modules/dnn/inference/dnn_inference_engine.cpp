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
 * Author: Ali Mansouri <ali.m.t1992@gmail.com>
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
using namespace ov;

namespace XCam {

DnnInferenceEngine::DnnInferenceEngine (DnnInferConfig& config)
    : _model_loaded (false)
    , _model_type (config.model_type)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::DnnInferenceEngine");
    _input_image_width.clear ();
    _input_image_height.clear ();

    layout_types["NCHW"] = ov_layout_nchw;
    layout_types["NHWC"] = ov_layout_nhwc;
    layout_types["OIHW"] = ov_layout_oihw;
    layout_types["C"] = ov_layout_c;
    layout_types["CHW"] = ov_layout_chw;
    layout_types["HW"] = ov_layout_hw;
    layout_types["NC"] = ov_layout_nc;
    layout_types["CN"] = ov_layout_cn;
    layout_types["BLOCKED"] = ov_layout_blocked;
    layout_types["ANY"] = ov_layout_any;
    
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
    return _ie->get_available_devices ();
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
    _ie = new ov::Core ();

    if ( (DnnInferDeviceCPU == config.target_id) && ("" != config.cpu_ext)) {
        XCAM_LOG_DEBUG ("Load CPU extensions %s", config.cpu_ext.c_str ());
        _ie->register_plugins(config.config_file.c_str ());
        // auto extensionPtr = std::make_shared<ov::Extension> (config.cpu_ext.c_str ());
        _ie->add_extension (config.cpu_ext.c_str ());
    } else if ((DnnInferDeviceGPU == config.target_id) && ("" != config.gpu_ext)) {
        XCAM_LOG_DEBUG ("Load GPU extensions: %s", config.gpu_ext.c_str ());
        _ie->register_plugins(config.config_file.c_str ());
        _ie->add_extension (config.gpu_ext.c_str ());
    } else if ((DnnInferDeviceHetero == config.target_id) && ("" != config.cpu_ext) && ("" != config.gpu_ext)) {
        _ie->register_plugins(config.config_file.c_str ());
        XCAM_LOG_DEBUG ("Load GPU extensions: %s", config.gpu_ext.c_str ());
        _ie->add_extension (config.cpu_ext.c_str ());
        XCAM_LOG_DEBUG ("Load CPU extensions: %s", config.cpu_ext.c_str ());
        _ie->add_extension (config.gpu_ext.c_str ());
        
        _ie->set_property ({ ov::device::priorities("GPU", "CPU") });
    }

    _network = _ie->read_model (get_filename_prefix (config.model_filename) + ".xml");

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

    ov::CompiledModel execute_network = _ie->compile_model (_network, config.device_name);

    _infer_request = execute_network.create_infer_request ();

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

    info.major = OPENVINO_VERSION_MAJOR;
    info.minor = OPENVINO_VERSION_MINOR;
    info.desc = get_openvino_version ().description;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_batch_size (const size_t size)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    ov::set_batch (_network, size);
    return XCAM_RETURN_NO_ERROR;
}

size_t
DnnInferenceEngine::get_batch_size ()
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return -1;
    }

    return ov::get_batch (_network).get_length ();
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
        _infer_request.infer ();
    } else {
        _infer_request.start_async ();
        _infer_request.wait ();
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

    return _network->inputs ().size ();
}

size_t
DnnInferenceEngine::get_output_size ()
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return -1;
    }

    return _network->get_output_size ();
}

XCamReturn
DnnInferenceEngine::set_input_precision (uint32_t idx, DnnInferPrecisionType precision)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (idx > get_input_size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ov::preprocess::PrePostProcessor ppp (_network);
    ov::preprocess::InputInfo& input_info = ppp.input (idx);

    ov::element::Type input_precision = convert_precision_type (precision);
    input_info.tensor().set_element_type (input_precision);
    _network = ppp.build();

    return XCAM_RETURN_NO_ERROR;
}

DnnInferPrecisionType
DnnInferenceEngine::get_input_precision (uint32_t idx)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return DnnInferPrecisionUndefined;
    }

    if (idx > get_input_size ()) {
        XCAM_LOG_ERROR ("Index is out of range");
        return DnnInferPrecisionUndefined;
    }

    ov::element::Type input_precision = _network->input (idx).get_element_type ();

    return convert_precision_type (input_precision);
}

XCamReturn
DnnInferenceEngine::set_output_precision (uint32_t idx, DnnInferPrecisionType precision)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (idx > get_output_size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ov::preprocess::PrePostProcessor ppp(_network);
    ov::preprocess::OutputInfo& output_info = ppp.output (idx);

    ov::element::Type output_precision = convert_precision_type (precision);
    output_info.tensor().set_element_type (output_precision);
    _network = ppp.build();

    return XCAM_RETURN_NO_ERROR;
}

DnnInferPrecisionType
DnnInferenceEngine::get_output_precision (uint32_t idx)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return DnnInferPrecisionUndefined;
    }

    if (idx > get_output_size ()) {
        XCAM_LOG_ERROR ("Index is out of range");
        return DnnInferPrecisionUndefined;
    }

    ov::element::Type output_precision = _network->output (idx).get_element_type ();

    return convert_precision_type (output_precision);
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

    if (idx > get_input_size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ov::preprocess::PrePostProcessor ppp (_network);
    ov::preprocess::InputInfo& input_info = ppp.input (idx);

    ov::Layout input_layout = convert_layout_type (layout);
    input_info.tensor().set_layout (input_layout);
    input_info.model().set_layout (input_layout);
    _network = ppp.build();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_output_layout (uint32_t idx, DnnInferLayoutType layout)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    if (idx > get_output_size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ov::preprocess::PrePostProcessor ppp(_network);
    ov::preprocess::OutputInfo& output_info = ppp.output (idx);

    ov::Layout output_layout = convert_layout_type (layout);
    output_info.tensor().set_layout (output_layout);
    output_info.model().set_layout (output_layout);
    _network = ppp.build();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_input_tensor (uint32_t idx, DnnInferData& data)
{
    if (NULL == _ie.ptr ()) {
        XCAM_LOG_ERROR ("Please create inference engine");
        return XCAM_RETURN_ERROR_ORDER;
    }

    std::string input_name;

    if (idx > get_input_size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    input_name = *(_network->input (idx).get_names ().begin ());

    if (input_name.empty ()) {
        XCAM_LOG_ERROR ("input name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (data.batch_idx > get_batch_size ()) {
        XCAM_LOG_ERROR ("Too many input, it is bigger than batch size!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ov::Tensor input_tensor = _infer_request.get_tensor (input_name);
    if (data.precision == DnnInferPrecisionFP32) {
        if (data.data_type == DnnInferDataTypeImage) {
            copy_image_to_input_tensor<element_type_traits<element::Type_t::f32>::value_type> (data, input_tensor, data.batch_idx);
        } else {
            copy_data_to_input_tensor<element_type_traits<element::Type_t::f32>::value_type> (data, input_tensor, data.batch_idx);
        }
    } else {
        if (data.data_type == DnnInferDataTypeImage) {
            copy_image_to_input_tensor<uint8_t>(data, input_tensor, data.batch_idx);
        } else {
            copy_data_to_input_tensor<uint8_t>(data, input_tensor, data.batch_idx);
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

        for (uint32_t index = 0; index > get_input_size (); index ++) {
            image_width = _network->input (index).get_shape ()[3];
            image_height = _network->input (index).get_shape ()[2];
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

            set_input_tensor (idx, image);
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

    for (VideoBufferList::const_iterator iter = images.begin(); iter != images.end (); ++iter) {
        SmartPtr<VideoBuffer> buf = *iter;
        XCAM_ASSERT (buf.ptr ());

        VideoBufferInfo buf_info = buf->get_video_info ();
        _input_image_width.push_back (buf_info.width);
        _input_image_height.push_back (buf_info.height);

        uint32_t image_width = 0;
        uint32_t image_height = 0;

        for (uint32_t index = 0; index < get_input_size (); index ++) {
            image_width = _network->input(index).get_shape()[3];
            image_height = _network->input(index).get_shape()[2];
        }

        float x_ratio = float(image_width) / float(buf_info.width);
        float y_ratio = float(image_height) / float(buf_info.height);

        uint8_t* data = NULL;
        if (buf_info.format == V4L2_PIX_FMT_NV12) {
            data = XCamDNN::convert_NV12_to_BGR (buf, x_ratio, y_ratio);
        } else if (buf_info.format == V4L2_PIX_FMT_BGR24) {
            data = XCamDNN::resize_BGR (buf, x_ratio, y_ratio);
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

            set_input_tensor (idx, image);
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

    if (index > get_output_size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    std::string output_name;

    for (uint32_t idx = 0; idx < get_output_size (); idx ++) {
        for (const auto & op : _network->get_ops ()) {
            if (op->get_type_info () == ngraph::op::DetectionOutput::get_type_info_static ()) {
                output_name = *(_network->output(idx).get_names ().begin ());
                break;
            }
        }
    }

    if (output_name.empty ()) {
        output_name = *(_network->output(0).get_names ().begin ());
    }

    if (output_name.empty ()) {
        XCAM_LOG_ERROR ("out name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    const ov::Tensor output_tensor = _infer_request.get_tensor (output_name);
    const auto output_data = static_cast<element_type_traits<element::Type_t::f32>::value_type*> (output_tensor.data ());

    size_t image_count = output_tensor.get_shape ()[0];
    size_t channels = output_tensor.get_shape ()[1];
    size_t image_height = output_tensor.get_shape ()[2];
    size_t image_width = output_tensor.get_shape ()[3];
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
    if (idx > get_output_size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return NULL;
    }

    if (NULL == _output_layer_type[_model_type]) {
        XCAM_LOG_ERROR ("Please set model output layer type!");
        return NULL;
    }

    for (uint32_t idx = 0; idx < get_output_size (); idx ++) {
        for (const auto & op : _network->get_ops ()) {
            if (op->get_type_info () == ngraph::op::DetectionOutput::get_type_info_static ()) {
                output_name = *(_network->output(idx).get_names ().begin ());
                break;
            }
        }
    }

    if (output_name.empty ()) {
        output_name = *(_network->output(0).get_names ().begin ());
    }

    if (output_name.empty ()) {
        XCAM_LOG_ERROR ("out name is empty!");
        return NULL;
    }

    const ov::Tensor output_tensor = _infer_request.get_tensor (output_name);
    float* output_result = static_cast<element_type_traits<element::Type_t::f32>::value_type*> (output_tensor.data ());

    size = output_tensor.get_byte_size ();

    return (reinterpret_cast<void *>(output_result));
}

ov::Layout
DnnInferenceEngine::estimate_layout_type (const int ch_num)
{
    if (ch_num == 4) {
        return ov::Layout("NCHW");
    } else if (ch_num == 3) {
        return ov::Layout("CHW");
    } else if (ch_num == 2) {
        return ov::Layout("NC");
    } else {
        return ov::Layout("ANY");
    }
}

ov::Layout
DnnInferenceEngine::convert_layout_type (DnnInferLayoutType layout)
{
    switch (layout) {
    case DnnInferLayoutNCHW:
        return ov::Layout("NCHW");
    case DnnInferLayoutNHWC:
        return ov::Layout("NHWC");
    case DnnInferLayoutOIHW:
        return ov::Layout("OIHW");
    case DnnInferLayoutC:
        return ov::Layout("C");
    case DnnInferLayoutCHW:
        return ov::Layout("CHW");
    case DnnInferLayoutHW:
        return ov::Layout("HW");
    case DnnInferLayoutNC:
        return ov::Layout("NC");
    case DnnInferLayoutCN:
        return ov::Layout("CN");
    case DnnInferLayoutBlocked:
        return ov::Layout("BLOCKED");
    case DnnInferLayoutAny:
        return ov::Layout("ANY");
    default:
        return ov::Layout("ANY");
    }
}

DnnInferLayoutType
DnnInferenceEngine::convert_layout_type (ov::Layout layout)
{
    switch (layout_types[layout.to_string ()]) {
    case ov_layout_nchw:
        return DnnInferLayoutNCHW;
    case ov_layout_nhwc:
        return DnnInferLayoutNHWC;
    case ov_layout_oihw:
        return DnnInferLayoutOIHW;
    case ov_layout_c:
        return DnnInferLayoutC;
    case ov_layout_chw:
        return DnnInferLayoutCHW;
    case ov_layout_hw:
        return DnnInferLayoutHW;
    case ov_layout_nc:
        return DnnInferLayoutNC;
    case ov_layout_cn:
        return DnnInferLayoutCN;
    case ov_layout_blocked:
        return DnnInferLayoutBlocked;
    case ov_layout_any:
        return DnnInferLayoutAny;
    default:
        return DnnInferLayoutAny;
    }
}

ov::element::Type
DnnInferenceEngine::convert_precision_type (DnnInferPrecisionType precision)
{
    switch (precision) {
    case DnnInferPrecisionU8:
        return ov::element::u8;
    case DnnInferPrecisionI8:
        return ov::element::i8;
    case DnnInferPrecisionU16:
        return ov::element::u16;
    case DnnInferPrecisionI16:
        return ov::element::i16;
    case DnnInferPrecisionFP16:
        return ov::element::f16;
    case DnnInferPrecisionI32:
        return ov::element::i32;
    case DnnInferPrecisionFP32:
        return ov::element::f32;
    case DnnInferPrecisionDynamic:
        return ov::element::dynamic;
    case DnnInferPrecisionUndefined:
        return ov::element::undefined;
    default:
        return ov::element::undefined;
    }
}

DnnInferPrecisionType
DnnInferenceEngine::convert_precision_type (ov::element::Type precision)
{
    switch (precision) {
    case ov::element::dynamic:
        return DnnInferPrecisionDynamic;
    case ov::element::f32:
        return DnnInferPrecisionFP32;
    case ov::element::f16:
        return DnnInferPrecisionFP16;
    case ov::element::i16:
        return DnnInferPrecisionI16;
    case ov::element::u8:
        return DnnInferPrecisionU8;
    case ov::element::i8:
        return DnnInferPrecisionI8;
    case ov::element::u16:
        return DnnInferPrecisionU16;
    case ov::element::i32:
        return DnnInferPrecisionI32;
    case ov::element::undefined:
        return DnnInferPrecisionUndefined;
    default:
        return DnnInferPrecisionUndefined;
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
DnnInferenceEngine::copy_image_to_input_tensor (const DnnInferData& data, ov::Tensor& image_tensor, int batch_index)
{
    // Filling input tensor with images. in order of b channel, then g and r channels

    size_t channels = image_tensor.get_shape()[1];
    const size_t image_width = image_tensor.get_shape()[3];
    const size_t image_height = image_tensor.get_shape()[2];
    size_t image_size =  image_width * image_height;

    T* tensor_data = image_tensor.data<T>();
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
            std::memcpy (tensor_data + batch_offset, buffer, image_size * channels);
        } else if (data.width == data.width_stride) {
            for (size_t ch = 0; ch < channels; ++ch) {
                std::memcpy (tensor_data + batch_offset + ch * image_size, buffer + ch * image_stride_size, image_size);
            }
        } else {
            for (size_t ch = 0; ch < channels; ch++) {
                for (size_t h = 0; h < image_height; h++) {
                    std::memcpy (tensor_data + batch_offset + ch * image_size + h * image_width, buffer + ch * image_stride_size + h * data.width_stride, image_width);
                }
            }
        }
    } else if (DnnInferImageFormatBGRPacked == data.image_format) {
        for (size_t pid = 0; pid < image_size; pid++) {
            for (size_t ch = 0; ch < channels; ch++) {
                tensor_data[batch_offset + ch * image_size + pid] = buffer[pid * channels + ch];
            }
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

template <typename T> XCamReturn
DnnInferenceEngine::copy_data_to_input_tensor (const DnnInferData& data, ov::Tensor& input_tensor, int batch_index)
{
    T * buffer = (T *)data.buffer;
    T* tensor_data = input_tensor.data<T>();

    int batch_offset = batch_index * data.size;

    memcpy (tensor_data + batch_offset, buffer, data.size);

    return XCAM_RETURN_NO_ERROR;
}

void
DnnInferenceEngine::print_performance_counts (const std::map<std::string, ov::ProfilingInfo>& performance_map)
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
        case ov::ProfilingInfo::Status::EXECUTED:
            XCAM_LOG_DEBUG ("EXECUTED");
            break;
        case ov::ProfilingInfo::Status::NOT_RUN:
            XCAM_LOG_DEBUG ("NOT_RUN");
            break;
        case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
            XCAM_LOG_DEBUG ("OPTIMIZED_OUT");
            break;
        }
        XCAM_LOG_DEBUG ("layerType: %s", std::string (it.second.node_type).c_str ());
        XCAM_LOG_DEBUG ("realTime: %d", it.second.real_time);
        XCAM_LOG_DEBUG ("cpu: %d", it.second.cpu_time);
        XCAM_LOG_DEBUG ("execType: %s", it.second.exec_type);
        if (it.second.real_time.count() > 0) {
            total_time += it.second.real_time.count();
        }
    }
    XCAM_LOG_DEBUG ("Total time: %d microseconds", total_time);
}

}  // namespace XCam
