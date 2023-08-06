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
 * Author: Ali Mansouri <ali.m.t1992@gmail.com>
 */

#ifndef XCAM_DNN_INFERENCE_ENGINE_H
#define XCAM_DNN_INFERENCE_ENGINE_H

#pragma once

#include <vector>
#include <string>
#include <openvino/openvino.hpp>

#include <xcam_std.h>
#include <video_buffer.h>

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
    DnnInferPrecisionDynamic,
    DnnInferPrecisionUndefined = -1
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

enum ov_layout_value {
    ov_layout_any = 0,
    ov_layout_nchw,
    ov_layout_nhwc,
    ov_layout_oihw,
    ov_layout_c,
    ov_layout_chw,
    ov_layout_hw,
    ov_layout_nc,
    ov_layout_cn,
    ov_layout_blocked
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
    DnnInferImageFormatNV12,
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

struct DnnInferImageSize {
    uint32_t image_width;
    uint32_t image_height;

    DnnInferImageSize () {
        image_width = 0;
        image_height = 0;
    }
};

struct DnnInferenceEngineInfo {
    int32_t major;
    int32_t minor;
    std::string desc;
    std::string name;

    DnnInferenceEngineInfo () {
        major = 0;
        minor = 0;
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

    std::string config_file;
    std::string device_name;
    std::string cpu_ext;
    std::string gpu_ext;
    std::string gna_ext;
    std::string model_filename;

    DnnInferConfig () {
        target_id = DnnInferDeviceCPU;
        device_name = "CPU";
    };
};

typedef std::map<DnnInferModelType, const char*> DnnOutputLayerType;
typedef std::map<std::string, ov_layout_value> OvLayoutType;

class DnnInferenceEngine {
public:
    explicit DnnInferenceEngine (DnnInferConfig& config);
    virtual ~DnnInferenceEngine ();

    XCamReturn create_model (DnnInferConfig& config);
    XCamReturn load_model (DnnInferConfig& config);

    std::vector<std::string> get_available_devices ();

    XCamReturn get_info (DnnInferenceEngineInfo& info);

    XCamReturn set_batch_size (const size_t size);
    size_t get_batch_size ();

    bool ready_to_start ()  const
    {
        return _model_loaded;
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

    uint32_t get_input_image_height (uint32_t idx) const {
        return (idx >= _input_image_height.size ()) ? 0 : _input_image_height[idx];
    };
    uint32_t get_input_image_width (uint32_t idx) const {
        return (idx >= _input_image_width.size ()) ? 0 : _input_image_width[idx];
    };

    virtual XCamReturn set_model_input_info (DnnInferInputOutputInfo& info) = 0;
    virtual XCamReturn get_model_input_info (DnnInferInputOutputInfo& info) = 0;

    virtual XCamReturn set_model_output_info (DnnInferInputOutputInfo& info) = 0;
    virtual XCamReturn get_model_output_info (DnnInferInputOutputInfo& info) = 0;

    virtual XCamReturn set_inference_data (std::vector<std::string> images);
    virtual XCamReturn set_inference_data (const VideoBufferList& images);

    void* get_inference_results (uint32_t idx, uint32_t& size);
    std::shared_ptr<uint8_t> read_input_image (std::string& image);
    XCamReturn save_output_image (const std::string& image_name, uint32_t index);

protected:
    virtual XCamReturn set_output_layer_type (const char* type) = 0;

    ov::Layout estimate_layout_type (const int ch_num);
    ov::Layout convert_layout_type (DnnInferLayoutType layout);
    DnnInferLayoutType convert_layout_type (ov::Layout layout);

    ov::element::Type convert_precision_type (DnnInferPrecisionType precision);
    DnnInferPrecisionType convert_precision_type (ov::element::Type precision);

    std::string get_filename_prefix (const std::string &file_path);

    void print_performance_counts (const std::map<std::string, ov::ProfilingInfo>& performance_map);

    XCamReturn set_input_tensor (uint32_t idx, DnnInferData& data);

private:
    template <typename T> XCamReturn copy_image_to_input_tensor (const DnnInferData& data, ov::Tensor& input_tensor, int batch_index);
    template <typename T> XCamReturn copy_data_to_input_tensor (const DnnInferData& data, ov::Tensor& input_tensor, int batch_index);

protected:

    bool _model_loaded;

    DnnInferModelType _model_type;

    std::vector<uint32_t> _input_image_width;
    std::vector<uint32_t> _input_image_height;

    SmartPtr<ov::Core> _ie;
    std::shared_ptr<ov::Model> _network;
    ov::InferRequest _infer_request;

    DnnOutputLayerType _output_layer_type;
    OvLayoutType layout_types;
};

}  // namespace XCam

#endif // XCAM_DNN_INFERENCE_ENGINE_H

