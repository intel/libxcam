/*
 * dnn_inference_engine.h -  dnn inference engine
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

#ifndef XCAM_DNN_INFERENCE_ENGINE_H
#define XCAM_DNN_INFERENCE_ENGINE_H

#pragma once

#include <vector>
#include <string>
#include <inference_engine.hpp>

#include <xcam_std.h>

namespace XCam {

enum DnnInferModelType {
    DnnInferObjectDetection = 0,
    DnnInferSemanticSegmentation,
    DnnInferSuperResolution,
    DnnInferHumanPoseEstimation,
    DnnInferTextDetection,
    DnnInferTextRecognition,
    DnnInferObjectRecognition
};

enum DnnInferTargetDeviceType {
    DnnInferDeviceDefault = 0,
    DnnInferDeviceBalanced,
    DnnInferDeviceCPU,
    DnnInferDeviceGPU,
    DnnInferDeviceFPGA,
    DnnInferDeviceMyriad,
    DnnInferDeviceHetero
};

enum DnnInferPrecisionType {
    DnnInferPrecisionU8 = 0,
    DnnInferPrecisionI8,
    DnnInferPrecisionU16,
    DnnInferPrecisionI16,
    DnnInferPrecisionQ78,
    DnnInferPrecisionFP16,
    DnnInferPrecisionI32,
    DnnInferPrecisionFP32,
    DnnInferPrecisionMixed,
    DnnInferPrecisionCustom,
    DnnInferPrecisionUnspecified = -1
};

enum DnnInferLayoutType {
    DnnInferLayoutAny = 0,
    DnnInferLayoutNCHW,
    DnnInferLayoutNHWC,
    DnnInferLayoutOIHW,
    DnnInferLayoutC,
    DnnInferLayoutCHW,
    DnnInferLayoutHW,
    DnnInferLayoutNC,
    DnnInferLayoutCN,
    DnnInferLayoutBlocked
};

enum DnnInferMemoryType {
    DnnInferMemoryDefault = 0,
    DnnInferMemoryHost,
    DnnInferMemoryGPU,
    DnnInferMemoryMYRIAD,
    DnnInferMemoryShared
};

enum DnnInferImageFormatType {
    DnnInferImageFormatBGRPacked = 0,
    DnnInferImageFormatBGRPlanar,
    DnnInferImageFormatRGBPacked,
    DnnInferImageFormatRGBPlanar,
    DnnInferImageFormatGrayPlanar,
    DnnInferImageFormatGeneric1D,
    DnnInferImageFormatGeneric2D,
    DnnInferImageFormatUnknown = -1
};

enum DnnInferMode {
    DnnInferModeSync = 0,
    DnnInferModeAsync
};

enum DnnInferDataType {
    DnnInferDataTypeNonImage = 0,
    DnnInferDataTypeImage
};

enum DnnInferLogLevel {
    DnnInferLogLevelNone = 0,
    DnnInferLogLevelEngine,
    DnnInferLogLevelLayer
};

enum DnnInferInfoType {
    DnnInferInfoEngine = 0,
    DnnInferInfoPlugin,
    DnnInferInfoNetwork
};

struct DnnInferImageSize {
    uint32_t image_width;
    uint32_t image_height;

    DnnInferImageSize () {
        image_width = 0;
        image_height = 0;
    }
};

struct DnnInferenceEngineInfo {
    DnnInferInfoType type;
    int32_t major;
    int32_t minor;
    const char* desc;
    const char* name;

    DnnInferenceEngineInfo () {
        type = DnnInferInfoEngine;
        major = 0;
        minor = 0;
        desc = NULL;
        name = NULL;
    };
};

#define DNN_INFER_MAX_INPUT_OUTPUT 10
struct DnnInferInputOutputInfo {
    uint32_t width[DNN_INFER_MAX_INPUT_OUTPUT];
    uint32_t height[DNN_INFER_MAX_INPUT_OUTPUT];
    uint32_t channels[DNN_INFER_MAX_INPUT_OUTPUT];
    uint32_t object_size[DNN_INFER_MAX_INPUT_OUTPUT];
    DnnInferPrecisionType precision[DNN_INFER_MAX_INPUT_OUTPUT];
    DnnInferLayoutType layout[DNN_INFER_MAX_INPUT_OUTPUT];
    DnnInferDataType data_type[DNN_INFER_MAX_INPUT_OUTPUT];
    DnnInferImageFormatType format[DNN_INFER_MAX_INPUT_OUTPUT];
    uint32_t batch_size;
    uint32_t numbers;
};

struct DnnInferData {
    void * buffer;
    uint32_t size;
    uint32_t width;
    uint32_t height;
    uint32_t width_stride;
    uint32_t height_stride;
    uint32_t channel_num;
    uint32_t batch_idx;
    DnnInferPrecisionType precision;
    DnnInferMemoryType mem_type;
    DnnInferImageFormatType image_format;
    DnnInferDataType data_type;

    DnnInferData () {
        buffer = NULL;
    };
};

struct DnnInferConfig {
    DnnInferModelType model_type;
    DnnInferTargetDeviceType target_id;
    DnnInferInputOutputInfo input_infos;
    DnnInferInputOutputInfo output_infos;

    char * plugin_path;
    char * cpu_ext_path;
    char * cldnn_ext_path;
    char * model_filename;
    char * output_layer_name;
    uint32_t perf_counter;
    uint32_t infer_req_num;

    DnnInferConfig () {
        plugin_path = NULL;
        cpu_ext_path = NULL;
        cldnn_ext_path = NULL;
        model_filename = NULL;
        output_layer_name = NULL;
    };
};

typedef std::map<DnnInferModelType, const char*> DnnOutputLayerType;

class DnnInferenceEngine {
public:
    explicit DnnInferenceEngine (DnnInferConfig& config);
    virtual ~DnnInferenceEngine ();

    XCamReturn create_model (DnnInferConfig& config);
    XCamReturn load_model (DnnInferConfig& config);

    XCamReturn get_info (DnnInferenceEngineInfo& info, DnnInferInfoType type);

    XCamReturn set_batch_size (const size_t size);
    size_t get_batch_size ();

    bool ready_to_start ()  const {
        return _model_created && _model_loaded;
    };

    XCamReturn start (bool sync = true);

    size_t get_input_size ();
    size_t get_output_size ();

    XCamReturn set_input_precision (uint32_t idx, DnnInferPrecisionType precision);
    DnnInferPrecisionType get_input_precision (uint32_t idx);
    XCamReturn set_output_precision (uint32_t idx, DnnInferPrecisionType precision);
    DnnInferPrecisionType get_output_precision (uint32_t idx);

    DnnInferImageFormatType get_output_format (uint32_t idx);

    XCamReturn set_input_layout (uint32_t idx, DnnInferLayoutType layout);
    XCamReturn set_output_layout (uint32_t idx, DnnInferLayoutType layout);

    uint32_t get_input_image_height () const {
        return _input_image_height;
    };
    uint32_t get_input_image_width () const {
        return _input_image_width;
    };

    virtual XCamReturn set_model_input_info (DnnInferInputOutputInfo& info) = 0;
    virtual XCamReturn get_model_input_info (DnnInferInputOutputInfo& info) = 0;

    virtual XCamReturn set_model_output_info (DnnInferInputOutputInfo& info) = 0;
    virtual XCamReturn get_model_output_info (DnnInferInputOutputInfo& info) = 0;

    virtual XCamReturn set_inference_data (std::vector<std::string> images);

    void* get_inference_results (uint32_t idx, uint32_t& size);
    std::shared_ptr<uint8_t> read_input_image (std::string& image);
    XCamReturn save_output_image (const std::string& image_name, uint32_t index);

    void print_log (uint32_t flag);

protected:
    virtual XCamReturn set_output_layer_type (const char* type) = 0;
    InferenceEngine::TargetDevice get_device_from_string (const std::string& device_name);
    InferenceEngine::TargetDevice get_device_from_id (DnnInferTargetDeviceType device);

    InferenceEngine::Layout estimate_layout_type (const int ch_num);
    InferenceEngine::Layout convert_layout_type (DnnInferLayoutType layout);
    DnnInferLayoutType convert_layout_type (InferenceEngine::Layout layout);

    InferenceEngine::Precision convert_precision_type (DnnInferPrecisionType precision);
    DnnInferPrecisionType convert_precision_type (InferenceEngine::Precision precision);

    std::string get_filename_prefix (const std::string &file_path);

    void print_performance_counts (const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performance_map);

    XCamReturn set_input_blob (uint32_t idx, DnnInferData& data);

private:
    template <typename T> XCamReturn copy_image_to_blob (const DnnInferData& data, InferenceEngine::Blob::Ptr& blob, int batch_index);
    template <typename T> XCamReturn copy_data_to_blob (const DnnInferData& data, InferenceEngine::Blob::Ptr& blob, int batch_index);

protected:

    bool _model_created;
    bool _model_loaded;

    DnnInferModelType _model_type;

    uint32_t _input_image_width;
    uint32_t _input_image_height;

    InferenceEngine::InferencePlugin _plugin;
    InferenceEngine::CNNNetReader _network_reader;
    InferenceEngine::CNNNetwork _network;
    InferenceEngine::InferRequest _infer_request;
    std::vector<InferenceEngine::CNNLayerPtr> _layers;

    DnnOutputLayerType _output_layer_type;
};

}  // namespace XCam

#endif // XCAM_DNN_INFERENCE_ENGINE_H

