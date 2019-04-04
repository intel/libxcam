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

#include <iomanip>

#include <format_reader_ptr.h>
#include <ext_list.hpp>

//#if HAVE_OPENCV
#include "ocv/cv_std.h"
//#endif

#include "dnn_inference_engine.h"

using namespace std;
using namespace InferenceEngine;

namespace XCam {

DnnInferenceEngine::DnnInferenceEngine (DnnInferConfig& config)
    : _model_created (false)
    , _model_loaded (false)
    , _model_type (config.model_type)
    , _input_image_width (0)
    , _input_image_height (0)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::DnnInferenceEngine");

    create_model (config);
}


DnnInferenceEngine::~DnnInferenceEngine ()
{

}

XCamReturn
DnnInferenceEngine::create_model (DnnInferConfig& config)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::create_model");
    if (_model_created) {
        XCAM_LOG_INFO ("model already created!");
        return XCAM_RETURN_NO_ERROR;
    }

    // 1. Read the Intermediate Representation
    XCAM_LOG_DEBUG ("pre-trained model file name: %s", config.model_filename);
    if (NULL == config.model_filename) {
        XCAM_LOG_ERROR ("Model file name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    _network_reader.ReadNetwork (get_filename_prefix (config.model_filename) + ".xml");
    _network_reader.ReadWeights (get_filename_prefix (config.model_filename) + ".bin");

    // 2. read network from model
    _network = _network_reader.getNetwork ();

    // 3. Select Plugin - Select the plugin on which to load your network.
    // 3.1. Create the plugin with the InferenceEngine::PluginDispatcher load helper class.
    if (NULL == config.plugin_path) {
        InferenceEngine::PluginDispatcher dispatcher ({""});
        _plugin = dispatcher.getPluginByDevice (getDeviceName (get_device_from_id (config.target_id)));
    } else {
        InferenceEngine::PluginDispatcher dispatcher ({config.plugin_path});
        _plugin = dispatcher.getPluginByDevice (getDeviceName (get_device_from_id (config.target_id)));
    }

    // 3.2. Pass per device loading configurations specific to this device,
    // and register extensions to this device.
    if (DnnInferDeviceCPU == config.target_id) {
        /**
        * cpu_extensions library is compiled from "extension" folder containing
        * custom MKLDNNPlugin layer implementations. These layers are not supported
        * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        _plugin.AddExtension (std::make_shared<Extensions::Cpu::CpuExtensions>());

        if (NULL != config.cpu_ext_path) {
            std::string cpu_ext_path (config.cpu_ext_path);
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            auto extensionPtr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(cpu_ext_path);
            _plugin.AddExtension (extensionPtr);
            XCAM_LOG_DEBUG ("CPU Extension loaded: %s", cpu_ext_path);
        }
    } else if (DnnInferDeviceGPU == config.target_id) {
        if (NULL != config.cldnn_ext_path) {
            std::string cldnn_ext_path (config.cldnn_ext_path);
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            _plugin.SetConfig ({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, cldnn_ext_path } });
            XCAM_LOG_DEBUG ("GPU Extension loaded: %s", cldnn_ext_path);
        }
    }

    if (config.perf_counter > 0) {
        _plugin.SetConfig ({ { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } });
    }

    _model_created = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::load_model (DnnInferConfig& config)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::load_model");
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }
    if (_model_loaded) {
        XCAM_LOG_INFO ("model already loaded!");
        return XCAM_RETURN_NO_ERROR;
    }

    InferenceEngine::ExecutableNetwork execute_network = _plugin.LoadNetwork (_network, {});

    _infer_request = execute_network.CreateInferRequest ();

    _model_loaded = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::get_info (DnnInferenceEngineInfo& info, DnnInferInfoType type)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::get_info type %d", type);

    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    info.type = type;
    if (DnnInferInfoEngine == type) {
        info.major = GetInferenceEngineVersion ()->apiVersion.major;
        info.minor = GetInferenceEngineVersion ()->apiVersion.minor;
    } else if (DnnInferInfoPlugin == type) {
        const InferenceEngine::Version *plugin_version = NULL;
        static_cast<InferenceEngine::InferenceEnginePluginPtr>(_plugin)->GetVersion (plugin_version);

        info.major = plugin_version->apiVersion.major;
        info.minor = plugin_version->apiVersion.minor;
        info.desc = plugin_version->description;
    } else if (DnnInferInfoNetwork == type) {
        info.major = _network_reader.getVersion ();
        info.desc = _network_reader.getDescription ().c_str ();
        info.name = _network_reader.getName ().c_str ();
    } else {
        XCAM_LOG_WARNING ("DnnInferenceEngine::get_info type %d not supported!", type);
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_batch_size (const size_t size)
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    _network.setBatchSize (size);
    return XCAM_RETURN_NO_ERROR;
}

size_t
DnnInferenceEngine::get_batch_size ()
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
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
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return -1;
    }

    InputsDataMap inputs_info (_network.getInputsInfo());
    return inputs_info.size ();
}

size_t
DnnInferenceEngine::get_output_size ()
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return -1;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo());
    return outputs_info.size ();
}

XCamReturn
DnnInferenceEngine::set_input_presion (uint32_t idx, DnnInferPrecisionType precision)
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    uint32_t id = 0;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    if (idx > inputs_info.size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (auto & item : inputs_info) {
        if (id == idx) {
            Precision input_precision = convert_precision_type (precision);
            item.second->setPrecision (input_precision);
            break;
        }
        id++;
    }

    return XCAM_RETURN_NO_ERROR;
}

DnnInferPrecisionType
DnnInferenceEngine::get_input_presion (uint32_t idx)
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return DnnInferPrecisionUnspecified;
    }

    uint32_t id = 0;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    if (idx > inputs_info.size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return DnnInferPrecisionUnspecified;
    }

    for (auto & item : inputs_info) {
        if (id == idx) {
            Precision input_precision = item.second->getPrecision ();
            return convert_precision_type (input_precision);
        }
        id++;
    }
    return DnnInferPrecisionUnspecified;
}

XCamReturn
DnnInferenceEngine::set_output_presion (uint32_t idx, DnnInferPrecisionType precision)
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    uint32_t id = 0;
    OutputsDataMap outputs_info (_network.getOutputsInfo ());

    if (idx > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (auto & item : outputs_info) {
        if (id == idx) {
            Precision output_precision = convert_precision_type (precision);
            item.second->setPrecision (output_precision);
            break;
        }
        id++;
    }

    return XCAM_RETURN_NO_ERROR;
}

DnnInferPrecisionType
DnnInferenceEngine::get_output_presion (uint32_t idx)
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return DnnInferPrecisionUnspecified;
    }

    uint32_t id = 0;
    OutputsDataMap outputs_info (_network.getOutputsInfo ());

    if (idx > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return DnnInferPrecisionUnspecified;
    }

    for (auto & item : outputs_info) {
        if (id == idx) {
            Precision output_precision = item.second->getPrecision ();
            return convert_precision_type (output_precision);
        }
        id++;
    }
    return DnnInferPrecisionUnspecified;
}

XCamReturn
DnnInferenceEngine::set_input_layout (uint32_t idx, DnnInferLayoutType layout)
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }
    uint32_t id = 0;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    if (idx > inputs_info.size ()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (auto & item : inputs_info) {
        if (id == idx) {
            /** Creating first input blob **/
            Layout input_layout = convert_layout_type (layout);
            item.second->setLayout (input_layout);
            break;
        }
        id++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_output_layout (uint32_t idx, DnnInferLayoutType layout)
{
    if (! _model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    uint32_t id = 0;
    OutputsDataMap outputs_info (_network.getOutputsInfo ());

    if (idx > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (auto & item : outputs_info) {
        if (id == idx) {
            Layout output_layout = convert_layout_type (layout);
            item.second->setLayout (output_layout);
            break;
        }
        id++;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DnnInferenceEngine::set_model_input_info (DnnInferInputOutputInfo& info)
{
    XCAM_LOG_DEBUG ("DnnInferenceEngine::set_model_input_info");

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
DnnInferenceEngine::set_model_output_info (DnnInferInputOutputInfo& info)
{
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo ());

    if (info.numbers != outputs_info.size()) {
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

XCamReturn
DnnInferenceEngine::set_input_blob (uint32_t idx, DnnInferData& data)
{
    unsigned int id = 0;
    std::string item_name;
    InputsDataMap inputs_info (_network.getInputsInfo ());

    if (idx > inputs_info.size()) {
        XCAM_LOG_ERROR ("Input is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (auto & item : inputs_info) {
        if (id == idx) {
            item_name = item.first;
            break;
        }
        id++;
    }

    if (item_name.empty ()) {
        XCAM_LOG_ERROR ("item name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (data.batch_idx > get_batch_size ()) {
        XCAM_LOG_ERROR ("Too many input, it is bigger than batch size!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    Blob::Ptr blob = _infer_request.GetBlob (item_name);
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
    if (!_model_created) {
        XCAM_LOG_ERROR ("Please create the model firstly!");
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

        _input_image_width = reader->width ();
        _input_image_height = reader->height ();

        uint32_t image_width = 0;
        uint32_t image_height = 0;

        for (auto & item : inputs_info) {
            image_width = inputs_info[item.first]->getDims()[0];
            image_height = inputs_info[item.first]->getDims()[1];
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
            image.precision = get_input_presion (idx);
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

std::shared_ptr<uint8_t>
DnnInferenceEngine::read_input_image (std::string image)
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
    if (! _model_created || ! _model_loaded) {
        XCAM_LOG_ERROR ("Please create and load the model firstly!");
        return XCAM_RETURN_ERROR_ORDER;
    }

    OutputsDataMap outputs_info (_network.getOutputsInfo ());
    if (index > outputs_info.size ()) {
        XCAM_LOG_ERROR ("Output is out of range");
        return XCAM_RETURN_ERROR_PARAM;
    }

    std::string model_type;
    std::string item_name;

    switch (_model_type) {
    case DnnInferObjectDetection :
        model_type = "DetectionOutput";
        break;
    case DnnInferSuperResolution :
        model_type = "Convolution";
        break;
    default :
        model_type = "DetectionOutput";
        break;
    }

    for (auto & item : outputs_info) {
        if (item.second->creatorLayer.lock ()->type == model_type.c_str ()) {
            item_name = item.first;
            break;
        }
    }

    if (item_name.empty ()) {
        XCAM_LOG_ERROR ("item name is empty!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    const Blob::Ptr output_blob = _infer_request.GetBlob (item_name);
    const auto output_data = output_blob->buffer ().as<PrecisionTrait<Precision::FP32>::value_type*> ();

    size_t image_count = output_blob->getTensorDesc ().getDims ()[0];
    size_t channels = output_blob->getTensorDesc ().getDims ()[1];
    size_t image_height = output_blob->getTensorDesc ().getDims ()[2];
    size_t image_width = output_blob->getTensorDesc ().getDims ()[3];
    size_t pixel_count = image_width * image_height;

    XCAM_LOG_DEBUG ("Output size [image count, channels, height, width]: %d, %d, %d, %d",
                    image_count, channels, image_height, image_width);

    if (index > image_count) {
        return XCAM_RETURN_ERROR_PARAM;
    }

#if HAVE_OPENCV
    std::vector<cv::Mat> image_planes {cv::Mat (image_height, image_width, CV_32FC1, &(output_data[index * pixel_count * channels + pixel_count * 2])),
                                       cv::Mat (image_height, image_width, CV_32FC1, &(output_data[index * pixel_count * channels + pixel_count])),
                                       cv::Mat (image_height, image_width, CV_32FC1, &(output_data[index * pixel_count * channels]))
                                      };

    for (auto & image : image_planes) {
        image.convertTo (image, CV_8UC1, 255);
    }
    cv::Mat result_image;
    cv::merge (image_planes, result_image);
    cv::imwrite (image_name.c_str (), result_image);
#else
    XCamDNN::save_bmp_file (image_name, output_data, image_width, image_height);
#endif

    return XCAM_RETURN_NO_ERROR;
}

InferenceEngine::TargetDevice
DnnInferenceEngine::get_device_from_string (const std::string &device_name)
{
    return InferenceEngine::TargetDeviceInfo::fromStr (device_name);
}

InferenceEngine::TargetDevice
DnnInferenceEngine::get_device_from_id (DnnInferTargetDeviceType device)
{
    switch (device) {
    case DnnInferDeviceDefault:
        return InferenceEngine::TargetDevice::eDefault;
    case DnnInferDeviceBalanced:
        return InferenceEngine::TargetDevice::eBalanced;
    case DnnInferDeviceCPU:
        return InferenceEngine::TargetDevice::eCPU;
    case DnnInferDeviceGPU:
        return InferenceEngine::TargetDevice::eGPU;
    case DnnInferDeviceFPGA:
        return InferenceEngine::TargetDevice::eFPGA;
    case DnnInferDeviceMyriad:
        return InferenceEngine::TargetDevice::eMYRIAD;
    case DnnInferDeviceHetero:
        return InferenceEngine::TargetDevice::eHETERO;
    default:
        return InferenceEngine::TargetDevice::eCPU;
    }
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
    case DnnInferPrecisionMixed:
        return InferenceEngine::Precision::MIXED;
    case DnnInferPrecisionFP32:
        return InferenceEngine::Precision::FP32;
    case DnnInferPrecisionFP16:
        return InferenceEngine::Precision::FP16;
    case DnnInferPrecisionQ78:
        return InferenceEngine::Precision::Q78;
    case DnnInferPrecisionI16:
        return InferenceEngine::Precision::I16;
    case DnnInferPrecisionU8:
        return InferenceEngine::Precision::U8;
    case DnnInferPrecisionI8:
        return InferenceEngine::Precision::I8;
    case DnnInferPrecisionU16:
        return InferenceEngine::Precision::U16;
    case DnnInferPrecisionI32:
        return InferenceEngine::Precision::I32;
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
DnnInferenceEngine::copy_image_to_blob (const DnnInferData& data, Blob::Ptr& blob, int batch_index)
{
    SizeVector blob_size = blob.get()->dims ();
    const size_t width = blob_size[0];
    const size_t height = blob_size[1];
    const size_t channels = blob_size[2];
    const size_t image_size = width * height;
    unsigned char* buffer = (unsigned char*)data.buffer;
    T* blob_data = blob->buffer ().as<T*>();

    if (width != data.width || height != data.height) {
        XCAM_LOG_ERROR ("Input Image size (%dx%d) is not matched with model required size (%dx%d)!",
                        data.width, data.height, width, height);
        return XCAM_RETURN_ERROR_PARAM;
    }

    int batch_offset = batch_index * height * width * channels;

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
            for (size_t ch = 0; ch < channels; ++ch) {
                for (size_t h = 0; h < height; h++) {
                    std::memcpy (blob_data + batch_offset + ch * image_size + h * width, buffer + ch * image_stride_size + h * data.width_stride, width);
                }
            }
        }
    } else if (DnnInferImageFormatBGRPacked == data.image_format) {
        for (size_t pid = 0; pid < image_size; pid++) {
            for (size_t ch = 0; ch < channels; ++ch) {
                blob_data[batch_offset + ch * image_size + pid] = buffer[pid * channels + ch];
            }
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

template <typename T> XCamReturn
DnnInferenceEngine::copy_data_to_blob (const DnnInferData& data, Blob::Ptr& blob, int batch_index)
{
    SizeVector blob_size = blob.get ()->dims ();
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

        std::cout << std::setw(max_layer_name) << std::left << to_print;
        switch (it.second.status) {
        case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
            std::cout << std::setw (15) << std::left << "EXECUTED";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
            std::cout << std::setw (15) << std::left << "NOT_RUN";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            std::cout << std::setw(15) << std::left << "OPTIMIZED_OUT";
            break;
        }
        std::cout << std::setw (30) << std::left << "layerType: " + std::string (it.second.layer_type) + " ";
        std::cout << std::setw (20) << std::left << "realTime: " + std::to_string (it.second.realTime_uSec);
        std::cout << std::setw (20) << std::left << " cpu: " + std::to_string (it.second.cpu_uSec);
        std::cout << " execType: " << it.second.exec_type << std::endl;
        if (it.second.realTime_uSec > 0) {
            total_time += it.second.realTime_uSec;
        }
    }
    std::cout << std::setw (20) << std::left << "Total time: " + std::to_string (total_time) << " microseconds" << std::endl;
}

void
DnnInferenceEngine::print_log (uint32_t flag)
{
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomance_map;

    if (flag == DnnInferLogLevelNone) {
        return;
    }

    if (flag & DnnInferLogLevelEngine) {
        static_cast<InferenceEngine::InferenceEnginePluginPtr>(_plugin)->GetPerformanceCounts (perfomance_map, NULL);
        print_performance_counts (perfomance_map);
    }

    if (flag & DnnInferLogLevelLayer) {
        perfomance_map = _infer_request.GetPerformanceCounts ();
        print_performance_counts (perfomance_map);
    }
}

}  // namespace XCam
