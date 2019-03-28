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
 *
 */

#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <time.h>

#include <xcam_std.h>
#include "test_common.h"

#include <dnn/dnn_inference_engine.h>

using namespace XCam;
using namespace InferenceEngine;

class Color {

public:
    Color (uint8_t r, uint8_t g, uint8_t b) {
        _red = r;
        _green = g;
        _blue = b;
    }

public:
    uint8_t _red;
    uint8_t _green;
    uint8_t _blue;
};

static void add_rectangles (
    uint8_t *data,  uint32_t width, uint32_t height,
    std::vector<int> rectangles, std::vector<int> classes, int thickness = 1)
{
    std::vector<Color> colors = {
        // colors to be used for bounding boxes
        Color ( 128, 64,  128 ),
        Color ( 232, 35,  244 ),
        Color ( 70,  70,  70 ),
        Color ( 156, 102, 102 ),
        Color ( 153, 153, 190 ),
        Color ( 153, 153, 153 ),
        Color ( 30,  170, 250 ),
        Color ( 0,   220, 220 ),
        Color ( 35,  142, 107 ),
        Color ( 152, 251, 152 ),
        Color ( 180, 130, 70 ),
        Color ( 60,  20,  220 ),
        Color ( 0,   0,   255 ),
        Color ( 142, 0,   0 ),
        Color ( 70,  0,   0 ),
        Color ( 100, 60,  0 ),
        Color ( 90,  0,   0 ),
        Color ( 230, 0,   0 ),
        Color ( 32,  11,  119 ),
        Color ( 0,   74,  111 ),
        Color ( 81,  0,   81 )
    };
    if (rectangles.size() % 4 != 0 || rectangles.size() / 4 != classes.size()) {
        return;
    }

    for (size_t i = 0; i < classes.size(); i++) {
        int x = rectangles.at(i * 4);
        int y = rectangles.at(i * 4 + 1);
        int w = rectangles.at(i * 4 + 2);
        int h = rectangles.at(i * 4 + 3);

        int cls = classes.at(i) % colors.size();  // color of a bounding box line

        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (w < 0) w = 0;
        if (h < 0) h = 0;

        if (static_cast<std::size_t>(x) >= width) {
            x = width - 1;
            w = 0;
            thickness = 1;
        }
        if (static_cast<std::size_t>(y) >= height) {
            y = height - 1;
            h = 0;
            thickness = 1;
        }

        if (static_cast<std::size_t>(x + w) >= width) {
            w = width - x - 1;
        }
        if (static_cast<std::size_t>(y + h) >= height) {
            h = height - y - 1;
        }

        thickness = std::min(std::min(thickness, w / 2 + 1), h / 2 + 1);

        size_t shift_first;
        size_t shift_second;
        for (int t = 0; t < thickness; t++) {
            shift_first = (y + t) * width * 3;
            shift_second = (y + h - t) * width * 3;
            for (int ii = x; ii < x + w + 1; ii++) {
                data[shift_first + ii * 3] = colors.at(cls)._red;
                data[shift_first + ii * 3 + 1] = colors.at(cls)._green;
                data[shift_first + ii * 3 + 2] = colors.at(cls)._blue;
                data[shift_second + ii * 3] = colors.at(cls)._red;
                data[shift_second + ii * 3 + 1] = colors.at(cls)._green;
                data[shift_second + ii * 3 + 2] = colors.at(cls)._blue;
            }
        }

        for (int t = 0; t < thickness; t++) {
            shift_first = (x + t) * 3;
            shift_second = (x + w - t) * 3;
            for (int ii = y; ii < y + h + 1; ii++) {
                data[shift_first + ii * width * 3] = colors.at(cls)._red;
                data[shift_first + ii * width * 3 + 1] = colors.at(cls)._green;
                data[shift_first + ii * width * 3 + 2] = colors.at(cls)._blue;
                data[shift_second + ii * width * 3] = colors.at(cls)._red;
                data[shift_second + ii * width * 3 + 1] = colors.at(cls)._green;
                data[shift_second + ii * width * 3 + 2] = colors.at(cls)._blue;
            }
        }
    }
}

static bool write_output_bmp (std::string name, unsigned char *data, uint32_t width, uint32_t height)
{
    std::ofstream out_file;
    out_file.open (name, std::ofstream::binary);
    if (!out_file.is_open ()) {
        return false;
    }

    unsigned char file[14] = {
        'B', 'M',           // magic
        0, 0, 0, 0,         // size in bytes
        0, 0,               // app data
        0, 0,               // app data
        40 + 14, 0, 0, 0      // start of data offset
    };
    unsigned char info[40] = {
        40, 0, 0, 0,        // info hd size
        0, 0, 0, 0,         // width
        0, 0, 0, 0,         // height
        1, 0,               // number color planes
        24, 0,              // bits per pixel
        0, 0, 0, 0,         // compression is none
        0, 0, 0, 0,         // image bits size
        0x13, 0x0B, 0, 0,   // horz resolution in pixel / m
        0x13, 0x0B, 0, 0,   // vert resolution (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
        0, 0, 0, 0,         // #colors in palette
        0, 0, 0, 0,         // #important colors
    };

    if (height > (size_t)std::numeric_limits<int32_t>::max ||
            width > (size_t)std::numeric_limits<int32_t>::max) {
        XCAM_LOG_ERROR ("File size is too big: %dx%d", height, width);
        return false;
    }

    int pad_size = static_cast<int>(4 - (width * 3) % 4) % 4;
    int size_data = static_cast<int>(width * height * 3 + height * pad_size);
    int size_all = size_data + sizeof(file) + sizeof(info);

    file[2] = (unsigned char)(size_all);
    file[3] = (unsigned char)(size_all >> 8);
    file[4] = (unsigned char)(size_all >> 16);
    file[5] = (unsigned char)(size_all >> 24);

    info[4] = (unsigned char)(width);
    info[5] = (unsigned char)(width >> 8);
    info[6] = (unsigned char)(width >> 16);
    info[7] = (unsigned char)(width >> 24);

    int32_t negative_height = -(int32_t)height;
    info[8] = (unsigned char)(negative_height);
    info[9] = (unsigned char)(negative_height >> 8);
    info[10] = (unsigned char)(negative_height >> 16);
    info[11] = (unsigned char)(negative_height >> 24);

    info[20] = (unsigned char)(size_data);
    info[21] = (unsigned char)(size_data >> 8);
    info[22] = (unsigned char)(size_data >> 16);
    info[23] = (unsigned char)(size_data >> 24);

    out_file.write(reinterpret_cast<char *>(file), sizeof(file));
    out_file.write(reinterpret_cast<char *>(info), sizeof(info));

    unsigned char pad[3] = { 0, 0, 0 };

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            unsigned char pixel[3];
            pixel[0] = data[y * width * 3 + x * 3];
            pixel[1] = data[y * width * 3 + x * 3 + 1];
            pixel[2] = data[y * width * 3 + x * 3 + 2];

            out_file.write(reinterpret_cast<char *>(pixel), 3);
        }
        out_file.write(reinterpret_cast<char *>(pad), pad_size);
    }
    return true;
}

static void usage (const char* arg0)
{
    printf ("Usage:\n"
            "%s --plugin PATH --input filename --model-name xx.xml ...\n"
            "\t--plugin            plugin path\n"
            "\t--target-dev        target device, default: DnnInferDeviceCPU\n"
            "\t--ext-path          extension path\n"
            "\t--model-file        model file name\n"
            "\t--input             input image \n"
            "\t--save              save output image \n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    const struct option long_opts[] = {
        {"plugin", required_argument, NULL, 'p'},
        {"target-dev", required_argument, NULL, 'd'},
        {"ext-path", required_argument, NULL, 'x'},
        {"model-file", required_argument, NULL, 'm'},
        {"input", required_argument, NULL, 'i'},
        {"save", required_argument, NULL, 's'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    // --------------------------- 0. Get inference engine configurations -----------------------------------------------------------
    XCAM_LOG_DEBUG ("0. Get inference engine configurations");

    DnnInferConfig infer_config;
    infer_config.target_id = DnnInferDeviceCPU;

    char* ext_path = NULL;
    char* input_image = NULL;
    bool save_output = true;

    int opt = -1;
    while ((opt = getopt_long (argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'p':
            XCAM_ASSERT (optarg);
            infer_config.plugin_path = optarg;
            break;
        case 'd':
            XCAM_ASSERT (optarg);
            infer_config.target_id = (DnnInferTargetDeviceType)(atoi (optarg));
            if (!strcasecmp (optarg, "CPU")) {
                infer_config.target_id = DnnInferDeviceCPU;
            } else if (!strcasecmp (optarg, "GPU")) {
                infer_config.target_id = DnnInferDeviceGPU;
            } else if (!strcasecmp (optarg, "FPGA")) {
                infer_config.target_id = DnnInferDeviceFPGA;
            } else if (!strcasecmp (optarg, "Myriad")) {
                infer_config.target_id = DnnInferDeviceMyriad;
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
            infer_config.model_filename = optarg;
            break;
        case 'i':
            XCAM_ASSERT (optarg);
            input_image = optarg;
            break;
        case 's':
            XCAM_ASSERT (optarg);
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'h':
            usage (argv[0]);
            return 0;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value: %c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    XCAM_LOG_DEBUG ("targetDevice: %d", infer_config.target_id );
    if (DnnInferDeviceCPU == infer_config.target_id) {
        infer_config.cpu_ext_path = ext_path;
        XCAM_LOG_DEBUG ("CPU Extension loaded: %s", infer_config.cpu_ext_path);
    } else if (DnnInferDeviceGPU == infer_config.target_id) {
        infer_config.cldnn_ext_path = ext_path;
        XCAM_LOG_DEBUG ("GPU Extension loaded: %s", infer_config.cldnn_ext_path);
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknown option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    printf ("plugin path:\t\t%s\n", (infer_config.plugin_path != NULL) ? infer_config.plugin_path : "NULL");
    printf ("target device id:\t\t%d\n", infer_config.target_id);
    printf ("extention path:\t\t%s\n", (ext_path != NULL) ? ext_path : "NULL");
    printf ("input image:\t\t%s\n", (input_image != NULL) ? input_image : "NULL");
    printf ("model file name:\t\t%s\n", (infer_config.model_filename != NULL) ? infer_config.model_filename : "NULL");

    // --------------------------- 1. Set input image file names -----------------------------------------------------------
    XCAM_LOG_DEBUG ("1. Set input image file names");
    std::vector<std::string> images;
    if (NULL != input_image) {
        images.push_back (input_image);
    }

    if (images.empty()) {
        XCAM_LOG_ERROR ("No suitable images were found!");
        return -1;
    }

    // --------------------------- 2. create inference engine --------------------------------------------------
    XCAM_LOG_DEBUG ("2. Create inference engine");
    infer_config.perf_counter = 0;

    SmartPtr<DnnInferenceEngine> infer_engine = new DnnInferenceEngine (infer_config);

    DnnInferenceEngineInfo infer_info;
    infer_engine->get_info (infer_info, DnnInferInfoEngine);
    XCAM_LOG_DEBUG ("Inference Engine version: %d.%d",  infer_info.major, infer_info.minor);

    infer_engine->get_info (infer_info, DnnInferInfoPlugin);
    XCAM_LOG_DEBUG ("Inference Engine plugin discription: %s",  infer_info.desc);
    XCAM_LOG_DEBUG ("Inference Engine plugin version: %d.%d",  infer_info.major, infer_info.minor);

    infer_engine->get_info (infer_info, DnnInferInfoNetwork);
    XCAM_LOG_DEBUG ("Inference Engine network name: %s",  infer_info.name);
    XCAM_LOG_DEBUG ("Inference Engine network discription: %s",  infer_info.desc);
    XCAM_LOG_DEBUG ("Inference Engine network version: %d.%d",  infer_info.major, infer_info.minor);

    // --------------------------- 3. Get model input infos --------------------------------------------------
    XCAM_LOG_DEBUG ("3. Get/Set model input infos");
    infer_engine->get_model_input_info (infer_config.input_infos);

    XCAM_LOG_DEBUG ("Input info :");
    for (uint32_t i = 0; i < infer_config.input_infos.numbers; i++) {
        infer_config.input_infos.data_type[i] = DnnInferDataTypeImage;
        infer_engine->set_input_presion (i, DnnInferPrecisionU8);
        XCAM_LOG_DEBUG ("Idx %d : [%d X %d X %d] , [%d %d %d], batch size = %d", i,
                        infer_config.input_infos.width[i], infer_config.input_infos.height[i], infer_config.input_infos.channels[i],
                        infer_config.input_infos.precision[i], infer_config.input_infos.layout[i], infer_config.input_infos.data_type[i],
                        infer_config.input_infos.batch_size);
    }

    // --------------------------- 4. Get model output infos -------------------------------------------------
    XCAM_LOG_DEBUG ("4. Get/Set model output infos");
    infer_engine->get_model_output_info (infer_config.output_infos);

    XCAM_LOG_DEBUG ("Output info :");
    for (uint32_t i = 0; i < infer_config.output_infos.numbers; i++) {
        infer_engine->set_output_presion (i, DnnInferPrecisionFP32);
        XCAM_LOG_DEBUG ("Idx %d : [%d X %d X %d] , [%d %d %d], batch size = %d", i,
                        infer_config.output_infos.width[i], infer_config.output_infos.height[i], infer_config.output_infos.channels[i],
                        infer_config.output_infos.precision[i], infer_config.output_infos.layout[i], infer_config.output_infos.data_type[i],
                        infer_config.output_infos.batch_size);
    }

    // --------------------------- 5. load inference model -------------------------------------------------
    XCAM_LOG_DEBUG ("5. load inference model");
    infer_engine->load_model (infer_config);

    // --------------------------- 6. Set inference data --------------------------------------------------------
    XCAM_LOG_DEBUG ("6. Set inference data");
    infer_engine->set_inference_data (images);

    // --------------------------- 7. Do inference ---------------------------------------------------------
    XCAM_LOG_DEBUG ("7. Start inference iterations");
    if (infer_engine->ready_to_start ()) {
        infer_engine->start ();
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
    std::vector<std::vector<int> > boxes(batch_size);
    std::vector<std::vector<int> > classes(batch_size);

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx ++) {
        result_ptr = (float*)infer_engine->get_inference_results (batch_idx, blob_size);
        if (NULL == result_ptr) {
            continue;
        }

        int image_width = infer_engine->get_input_image_width ();
        int image_height = infer_engine->get_input_image_height ();
        int max_proposal_count = infer_config.output_infos.channels[batch_idx];
        int object_size = infer_config.output_infos.object_size[batch_idx];

        for (int32_t cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
            float image_id = result_ptr[cur_proposal * object_size + 0];
            if (image_id < 0) {
                break;
            }

            float label = result_ptr[cur_proposal * object_size + 1];
            float confidence = result_ptr[cur_proposal * object_size + 2];
            float xmin = result_ptr[cur_proposal * object_size + 3] * image_width;
            float ymin = result_ptr[cur_proposal * object_size + 4] * image_height;
            float xmax = result_ptr[cur_proposal * object_size + 5] * image_width;
            float ymax = result_ptr[cur_proposal * object_size + 6] * image_height;

            if (confidence > 0.5) {
                classes[image_id].push_back(static_cast<int>(label));
                boxes[image_id].push_back(static_cast<int>(xmin));
                boxes[image_id].push_back(static_cast<int>(ymin));
                boxes[image_id].push_back(static_cast<int>(xmax - xmin));
                boxes[image_id].push_back(static_cast<int>(ymax - ymin));

                XCAM_LOG_DEBUG ("Proposal:%d label:%d confidence:%f", cur_proposal, (int)label, confidence);
                XCAM_LOG_DEBUG ("Boxes[%f] {%d, %d, %d, %d}",
                                image_id, (int)xmin, (int)ymin, (int)xmax, (int)ymax);
            }
        }

        if (save_output) {
            std::shared_ptr<unsigned char> orig_image = infer_engine->read_inference_image (images[batch_idx]);
            add_rectangles (orig_image.get (),
                            image_width, image_height,
                            boxes[batch_idx], classes[batch_idx]);

            const std::string image_path = images[batch_idx] + "_out_" + std::to_string (batch_idx) + ".bmp";
            if (write_output_bmp (image_path, orig_image.get (), image_width, image_height)) {
                XCAM_LOG_DEBUG ("Image %s created!", image_path.c_str ());
            } else {
                XCAM_LOG_ERROR ("Can't create image file: %s", image_path.c_str ());
            }
        }
    }

    XCAM_LOG_DEBUG ("Execution successful");
    return 0;
}
