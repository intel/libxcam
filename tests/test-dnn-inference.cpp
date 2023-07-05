/*
 * test-dnn-inference.cpp -  test dnn inference sample
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
 *
 */

#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <time.h>

#include <xcam_std.h>
#include <soft/soft_video_buf_allocator.h>
#include "test_common.h"
#include "test_stream.h"

#include "dnn/inference/dnn_inference_utils.h"
#include "dnn/inference/dnn_inference_engine.h"
#include "dnn/inference/dnn_object_detection.h"
#include "dnn/inference/dnn_super_resolution.h"
#include "dnn/inference/dnn_semantic_segmentation.h"

using namespace XCam;
using namespace ov;

class InferStream
    : public Stream
{
public:
    explicit InferStream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~InferStream () {}

    virtual XCamReturn create_buf_pool (uint32_t reserve_count, uint32_t format = V4L2_PIX_FMT_BGR24);

private:
    XCAM_DEAD_COPY (InferStream);
};

typedef std::vector<SmartPtr<InferStream>> InferStreams;

InferStream::InferStream (const char *file_name, uint32_t width, uint32_t height)
    :  Stream (file_name, width, height)
{
}

XCamReturn
InferStream::create_buf_pool (uint32_t reserve_count, uint32_t format)
{
    XCAM_ASSERT (get_width () && get_height ());

    VideoBufferInfo info;
    info.init (format, get_width (), get_height ());

    SmartPtr<BufferPool> pool = new SoftVideoBufAllocator (info);
    XCAM_ASSERT (pool.ptr ());

    if (!pool->reserve (reserve_count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        pool.release ();
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

static void
write_out_image (const SmartPtr<InferStream> &out)
{
    out->write_buf ();
}

static void usage (const char* arg0)
{
    printf ("Usage:\n"
            "%s --input filename --model detect --model-file xx.xml ...\n"
            "\t--plugin            plugin path\n"
            "\t--target-dev        target device, default: CPU\n"
            "\t                    selected from: CPU, GPU\n"
            "\t--ext-path          extension path\n"
            "\t--model             pre-trained model name\n"
            "\t                    selected from: Detect, SR, Segment\n"
            "\t--model-file        model file name\n"
            "\t--input-image       input image\n"
            "\t--input-video       input video\n"
            "\t--in-w              optional, input width, default: 1280\n"
            "\t--in-h              optional, input height, default: 800\n"
            "\t--output            output image(NV12/MP4)\n"
            "\t--save              save output image\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    const struct option long_opts[] = {
        {"target-dev", required_argument, NULL, 'd'},
        {"ext-path", required_argument, NULL, 'x'},
        {"model", required_argument, NULL, 'm'},
        {"model-file", required_argument, NULL, 'f'},
        {"input-image", required_argument, NULL, 'i'},
        {"input-video", required_argument, NULL, 'v'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"output", required_argument, NULL, 'o'},
        {"save", required_argument, NULL, 's'},
        {"help", no_argument, NULL, 'H'},
        {NULL, 0, NULL, 0},
    };

    // --------------------------- 0. Get inference engine configurations -----------------------------------------------------------
    XCAM_LOG_DEBUG ("0. Get inference engine configurations");

    InferStreams ins;
    InferStreams outs;
    uint32_t input_width = 672;
    uint32_t input_height = 384;

    DnnInferConfig infer_config;
    infer_config.target_id = DnnInferDeviceCPU;

    char* ext_path = NULL;
    char* input_image = NULL;
    std::vector<std::string> images;
    bool save_output = true;
    bool process_video = false;

    int32_t opt = -1;
    while ((opt = getopt_long (argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'd':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "CPU")) {
                infer_config.target_id = DnnInferDeviceCPU;
            } else if (!strcasecmp (optarg, "GPU")) {
                infer_config.target_id = DnnInferDeviceGPU;
            } else {
                XCAM_LOG_ERROR ("target device unknown type: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'x':
            XCAM_ASSERT (optarg);
            ext_path = optarg;
            break;
        case 'm':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "Detect")) {
                infer_config.model_type = DnnInferObjectDetection;
            } else if (!strcasecmp (optarg, "Segment")) {
                infer_config.model_type = DnnInferSemanticSegmentation;
            } else if (!strcasecmp (optarg, "SR")) {
                infer_config.model_type = DnnInferSuperResolution;
            } else {
                XCAM_LOG_ERROR ("unsupported model type: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'f':
            XCAM_ASSERT (optarg);
            infer_config.model_filename = optarg;
            break;
        case 'i':
            XCAM_ASSERT (optarg);
            input_image = optarg;
            break;
        case 'v':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (InferStream, ins, optarg);
            break;
        case 'w':
            XCAM_ASSERT (optarg);
            input_width = (uint32_t)atoi(optarg);
            break;
        case 'h':
            XCAM_ASSERT (optarg);
            input_height = (uint32_t)atoi(optarg);
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (InferStream, outs, optarg);
            break;
        case 's':
            XCAM_ASSERT (optarg);
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'H':
            usage (argv[0]);
            return 0;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value: %c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (DnnInferDeviceCPU == infer_config.target_id) {
        infer_config.device_name = "CPU";
        if (NULL != ext_path ) {
            infer_config.cpu_ext = ext_path;
        }
        XCAM_LOG_DEBUG ("Device(%s) load extension: %s", infer_config.device_name.c_str (), infer_config.cpu_ext.c_str ());
    } else if (DnnInferDeviceGPU == infer_config.target_id) {
        infer_config.device_name = "GPU";
        if (NULL != ext_path ) {
            infer_config.gpu_ext = ext_path;
        }
        XCAM_LOG_DEBUG ("Device(%s) load extension: %s", infer_config.device_name.c_str (), infer_config.gpu_ext.c_str ());
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknown option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    printf ("target device id:\t\t%d\n", infer_config.target_id);
    printf ("extention path:\t\t%s\n", (ext_path != NULL) ? ext_path : "NULL");
    printf ("input image:\t\t%s\n", (input_image != NULL) ? input_image : "NULL");
    printf ("model type:\t\t%d\n", infer_config.model_type);
    printf ("model file name:\t\t%s\n", (infer_config.model_filename.c_str () != NULL) ? infer_config.model_filename.c_str () : "NULL");

    // --------------------------- 1. Set input image file names -----------------------------------------------------------
    if (ins.size () == 1 && ins[0].ptr ()) {
        process_video = true;
        ins[0]->set_buf_size (input_width, input_height);
        CHECK (ins[0]->create_buf_pool (6, V4L2_PIX_FMT_BGR24), "create buffer pool failed");
        CHECK (ins[0]->open_reader ("rb"), "open input file(%s) failed", ins[0]->get_file_name ());

        if (save_output && outs.size () == 1) {
            outs[0]->set_buf_size (input_width, input_height);
            CHECK (outs[0]->estimate_file_format (),
                   "%s: estimate file format failed", outs[0]->get_file_name ());
            CHECK (outs[0]->open_writer ("wb"), "open output file(%s) failed", outs[0]->get_file_name ());
        }
    } else {
        XCAM_LOG_DEBUG ("1. Set input image file names");
        if (NULL != input_image) {
            images.push_back (input_image);
        }

        if (images.empty()) {
            XCAM_LOG_ERROR ("No suitable images were found!");
            return -1;
        }
    }

    // --------------------------- 2. create inference engine --------------------------------------------------
    XCAM_LOG_DEBUG ("2. Create inference engine");
    SmartPtr<DnnInferenceEngine> infer_engine;

    if (DnnInferObjectDetection == infer_config.model_type) {
        infer_engine = new DnnObjectDetection (infer_config);
    } else if (DnnInferSuperResolution == infer_config.model_type) {
        infer_engine = new DnnSuperResolution (infer_config);
    } else if (DnnInferSemanticSegmentation == infer_config.model_type) {
        infer_engine = new DnnSemanticSegmentation (infer_config);
    } else {
        XCAM_LOG_ERROR ("Unsupported model type!");
        return -1;
    }

    std::vector<std::string> devices = infer_engine->get_available_devices ();
    XCAM_LOG_DEBUG ("Available target devices count: %d", devices.size ());
    for (const auto& it : devices) {
        XCAM_LOG_DEBUG ("  %s", it.c_str ());
    }

    DnnInferenceEngineInfo infer_info;
    CHECK (
        infer_engine->get_info (infer_info),
        "get inference engine info failed!");
    XCAM_LOG_DEBUG ("Inference Engine discription: %s",  infer_info.desc.c_str ());
    XCAM_LOG_DEBUG ("Inference Engine version: %d.%d",  infer_info.major, infer_info.minor);

    // --------------------------- 3. Get model input infos --------------------------------------------------
    XCAM_LOG_DEBUG ("3. Get/Set model input infos");
    CHECK (
        infer_engine->get_model_input_info (infer_config.input_infos),
        "get model input info failed!");

    XCAM_LOG_DEBUG ("Input info :");
    for (uint32_t i = 0; i < infer_config.input_infos.numbers; i++) {
        infer_config.input_infos.data_type[i] = DnnInferDataTypeImage;
        CHECK (
            infer_engine->set_input_precision (i, DnnInferPrecisionU8),
            "set input presion failed!");
        XCAM_LOG_DEBUG ("Idx %d : [%d X %d X %d] , [%d %d %d], batch size = %d", i,
                        infer_config.input_infos.width[i], infer_config.input_infos.height[i], infer_config.input_infos.channels[i],
                        infer_config.input_infos.precision[i], infer_config.input_infos.layout[i], infer_config.input_infos.data_type[i],
                        infer_config.input_infos.batch_size);
    }

    // --------------------------- 4. Get model output infos -------------------------------------------------
    XCAM_LOG_DEBUG ("4. Get/Set model output infos");
    CHECK (
        infer_engine->get_model_output_info (infer_config.output_infos),
        "get model output info failed!");
    XCAM_LOG_DEBUG ("Output info (numbers %d) :", infer_config.output_infos.numbers);

    for (uint32_t i = 0; i < infer_config.output_infos.numbers; i++) {
        CHECK (
            infer_engine->set_output_precision (i, DnnInferPrecisionFP32),
            "set output presion failed!");
        XCAM_LOG_DEBUG ("Idx %d : [%d X %d X %d] , [%d %d %d], batch size = %d", i,
                        infer_config.output_infos.width[i],
                        infer_config.output_infos.height[i],
                        infer_config.output_infos.channels[i],
                        infer_config.output_infos.precision[i],
                        infer_config.output_infos.layout[i],
                        infer_config.output_infos.data_type[i],
                        infer_config.output_infos.batch_size);
    }

    // --------------------------- 5. load inference model -------------------------------------------------
    XCAM_LOG_DEBUG ("5. load inference model");
    CHECK (
        infer_engine->load_model (infer_config),
        "load model failed!");

    if (process_video && DnnInferObjectDetection == infer_config.model_type) {
        do {
            if (ins[0]->read_buf() == XCAM_RETURN_BYPASS)
                break;

            VideoBufferList detect_buffers;
            detect_buffers.clear ();
            detect_buffers.push_back (ins[0]->get_buf ());
            infer_engine->set_inference_data (detect_buffers);

            if (infer_engine->ready_to_start ()) {
                CHECK (
                    infer_engine->start (),
                    "inference failed!");
            }

            size_t batch_size = infer_engine->get_output_size ();
            uint32_t blob_size = 0;
            float* result_ptr = NULL;

            for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx ++) {
                result_ptr = (float*)infer_engine->get_inference_results (batch_idx, blob_size);
                if (NULL == result_ptr) {
                    continue;
                }

                std::vector<Vec4i> boxes;
                std::vector<int32_t> classes;
                uint32_t image_width = infer_engine->get_input_image_width (batch_idx);
                uint32_t image_height = infer_engine->get_input_image_height (batch_idx);

                SmartPtr<DnnObjectDetection> object_detector = infer_engine.dynamic_cast_ptr<DnnObjectDetection> ();
                CHECK (
                    object_detector->get_bounding_boxes (result_ptr, batch_idx, boxes, classes),
                    "get bounding box failed!");

                outs[0]->get_buf () = detect_buffers.front ();
                uint8_t* detect_image = outs[0]->get_buf ()->map ();
                CHECK (
                    XCamDNN::draw_bounding_boxes (detect_image,
                                                  image_width, image_height, DnnInferImageFormatRGBPacked,
                                                  boxes, classes),
                    "Draw bounding boxes failed!" );

                outs[0]->get_buf ()->unmap ();

                if (save_output) {
                    write_out_image (outs[0]);
                }
            }
        } while (true);
    } else {
        // --------------------------- 6. Set inference data --------------------------------------------------------
        XCAM_LOG_DEBUG ("6. Set inference data");
        CHECK (
            infer_engine->set_inference_data (images),
            "set inference data failed!");

        // --------------------------- 7. Do inference ---------------------------------------------------------
        XCAM_LOG_DEBUG ("7. Start inference iterations");
        if (infer_engine->ready_to_start ()) {
            CHECK (
                infer_engine->start (),
                "inference failed!");
        }

        FPS_CALCULATION (inference_engine, XCAM_OBJ_DUR_FRAME_NUM);

        // --------------------------- 8. Process inference results -------------------------------------------------------
        XCAM_LOG_DEBUG ("Processing inference results");

        size_t batch_size = infer_engine->get_output_size ();
        if (batch_size != images.size ()) {
            XCAM_LOG_DEBUG ( "Number of images: %d ", images.size ());
            batch_size = std::min(batch_size, images.size ());
            XCAM_LOG_DEBUG ( "Number of images to be processed is: %d ", batch_size);
        }

        uint32_t blob_size = 0;
        float* result_ptr = NULL;

        for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx ++) {
            result_ptr = (float*)infer_engine->get_inference_results (batch_idx, blob_size);
            if (NULL == result_ptr) {
                continue;
            }

            if (DnnInferObjectDetection == infer_config.model_type) {
                std::vector<Vec4i> boxes;
                std::vector<int32_t> classes;
                uint32_t image_width = infer_engine->get_input_image_width (batch_idx);
                uint32_t image_height = infer_engine->get_input_image_height (batch_idx);

                SmartPtr<DnnObjectDetection> object_detector = infer_engine.dynamic_cast_ptr<DnnObjectDetection> ();
                CHECK (
                    object_detector->get_bounding_boxes (result_ptr, batch_idx, boxes, classes),
                    "get bounding box failed!");

                if (save_output) {
                    std::shared_ptr<unsigned char> input_image = infer_engine->read_input_image (images[batch_idx]);

                    CHECK (
                        XCamDNN::draw_bounding_boxes (input_image.get (),
                                                      image_width, image_height, DnnInferImageFormatRGBPacked,
                                                      boxes, classes),
                        "Draw bounding boxes failed!" );

                    const std::string image_path = images[batch_idx] + "_obj_detect_out_" + std::to_string (batch_idx) + ".bmp";

                    CHECK (
                        XCamDNN::save_bmp_file (image_path,
                                                input_image.get (),
                                                DnnInferImageFormatRGBPacked,
                                                DnnInferPrecisionU8,
                                                image_width,
                                                image_height),
                        "Can't create image file: %s",
                        image_path.c_str () );
                    XCAM_LOG_DEBUG ("Image %s created!", image_path.c_str ());

                }
            } else if (DnnInferSuperResolution == infer_config.model_type) {
                if (save_output) {
                    const std::string image_path = images[batch_idx] + "_super_res_out_" + std::to_string (batch_idx) + ".bmp";

                    CHECK (
                        infer_engine->save_output_image (image_path, batch_idx),
                        "Can't create image file: %s",
                        image_path.c_str () );
                    XCAM_LOG_DEBUG ("Image %s created!", image_path.c_str ());
                }
            }  else if (DnnInferSemanticSegmentation == infer_config.model_type) {
                SmartPtr<DnnSemanticSegmentation> semantic_seg = infer_engine.dynamic_cast_ptr<DnnSemanticSegmentation> ();
                if (save_output) {
                    const std::string image_path = images[batch_idx] + "_semantic_seg_out_" + std::to_string (batch_idx) + ".bmp";
                    uint32_t map_width = infer_config.output_infos.width[batch_idx];
                    uint32_t map_height = infer_config.output_infos.height[batch_idx];

                    std::vector<std::vector<uint32_t>> seg_map (map_height, std::vector<uint32_t>(map_width, 0));
                    CHECK (
                        semantic_seg->get_segmentation_map (result_ptr, batch_idx, seg_map),
                        "get segmentation map failed!");

                    CHECK (
                        XCamDNN::label_pixels (image_path, seg_map),
                        "Can't create image file: %s",
                        image_path.c_str () );
                    XCAM_LOG_DEBUG ("Image %s created!", image_path.c_str ());
                }
            }
        }
    }
    XCAM_LOG_DEBUG ("Execution successful");
    return 0;
}
