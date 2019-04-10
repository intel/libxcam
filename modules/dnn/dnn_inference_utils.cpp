/*
 * dnn_inference_utils.cpp -  dnn inference utils
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
#include <fstream>
#include <limits>

#include <xcam_std.h>
#include <vec_mat.h>

#include "dnn_inference_utils.h"

using namespace std;
using namespace XCam;

namespace XCamDNN {

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

static std::vector<Color> colors = {
    {128, 64,  128},
    {232, 35,  244},
    {70,  70,  70},
    {156, 102, 102},
    {153, 153, 190},
    {153, 153, 153},
    {30,  170, 250},
    {0,   220, 220},
    {35,  142, 107},
    {152, 251, 152},
    {180, 130, 70},
    {60,  20,  220},
    {0,   0,   255},
    {142, 0,   0},
    {70,  0,   0},
    {100, 60,  0},
    {90,  0,   0},
    {230, 0,   0},
    {32,  11,  119},
    {0,   74,  111},
    {81,  0,   81}
};

static unsigned char file_header[14] = {
    'B', 'M',           // magic
    0, 0, 0, 0,         // size in bytes
    0, 0,               // app data
    0, 0,               // app data
    40 + 14, 0, 0, 0    // start of data offset
};

static unsigned char header_info[40] = {
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

static inline void
clamp (float &value, float min, float max)
{
    value = (value < min) ? min : ((value > max) ? max : value);
}

XCamReturn
draw_bounding_boxes (
    uint8_t *data, uint32_t width, uint32_t height,
    std::vector<Vec4i> rectangles, std::vector<int32_t> classes, int32_t thickness)
{
    if (rectangles.size() != classes.size()) {
        return XCAM_RETURN_ERROR_PARAM;
    }

    for (size_t i = 0; i < classes.size(); i++) {
        int32_t x = rectangles.at(i)[0];
        int32_t y = rectangles.at(i)[1];
        int32_t w = rectangles.at(i)[2];
        int32_t h = rectangles.at(i)[3];

        // color of a bounding box line
        int32_t cls = classes.at(i) % colors.size();

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
        for (int32_t t = 0; t < thickness; t++) {
            shift_first = (y + t) * width * 3;
            shift_second = (y + h - t) * width * 3;
            for (int32_t ii = x; ii < x + w + 1; ii++) {
                data[shift_first + ii * 3] = colors.at(cls)._red;
                data[shift_first + ii * 3 + 1] = colors.at(cls)._green;
                data[shift_first + ii * 3 + 2] = colors.at(cls)._blue;
                data[shift_second + ii * 3] = colors.at(cls)._red;
                data[shift_second + ii * 3 + 1] = colors.at(cls)._green;
                data[shift_second + ii * 3 + 2] = colors.at(cls)._blue;
            }
        }

        for (int32_t t = 0; t < thickness; t++) {
            shift_first = (x + t) * 3;
            shift_second = (x + w - t) * 3;
            for (int32_t ii = y; ii < y + h + 1; ii++) {
                data[shift_first + ii * width * 3] = colors.at(cls)._red;
                data[shift_first + ii * width * 3 + 1] = colors.at(cls)._green;
                data[shift_first + ii * width * 3 + 2] = colors.at(cls)._blue;
                data[shift_second + ii * width * 3] = colors.at(cls)._red;
                data[shift_second + ii * width * 3 + 1] = colors.at(cls)._green;
                data[shift_second + ii * width * 3 + 2] = colors.at(cls)._blue;
            }
        }
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
label_pixels (const std::string name, std::vector<std::vector<uint32_t>> map)
{
    std::ofstream out_file;
    out_file.open (name, std::ofstream::binary);
    if (!out_file.is_open ()) {
        return XCAM_RETURN_ERROR_FILE;
    }

    auto height = map.size();
    auto width = map.at(0).size();

    if (height > (size_t)std::numeric_limits<int32_t>::max ||
            width > (size_t)std::numeric_limits<int32_t>::max) {
        XCAM_LOG_ERROR ("File size is too big: %dx%d", height, width);
        return XCAM_RETURN_ERROR_PARAM;
    }

    int32_t pad_size = static_cast<int32_t>(4 - (width * 3) % 4) % 4;
    int32_t size_data = static_cast<int32_t>(width * height * 3 + height * pad_size);
    int32_t size_all = size_data + sizeof(file_header) + sizeof(header_info);

    file_header[2] = (unsigned char)(size_all);
    file_header[3] = (unsigned char)(size_all >> 8);
    file_header[4] = (unsigned char)(size_all >> 16);
    file_header[5] = (unsigned char)(size_all >> 24);

    header_info[4] = (unsigned char)(width);
    header_info[5] = (unsigned char)(width >> 8);
    header_info[6] = (unsigned char)(width >> 16);
    header_info[7] = (unsigned char)(width >> 24);

    int32_t negative_height = -(int32_t)height;
    header_info[8] = (unsigned char)(negative_height);
    header_info[9] = (unsigned char)(negative_height >> 8);
    header_info[10] = (unsigned char)(negative_height >> 16);
    header_info[11] = (unsigned char)(negative_height >> 24);

    header_info[20] = (unsigned char)(size_data);
    header_info[21] = (unsigned char)(size_data >> 8);
    header_info[22] = (unsigned char)(size_data >> 16);
    header_info[23] = (unsigned char)(size_data >> 24);

    out_file.write (reinterpret_cast<char *>(file_header), sizeof(file_header));
    out_file.write (reinterpret_cast<char *>(header_info), sizeof(header_info));

    unsigned char pad[3] = {0, 0, 0};

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            unsigned char pixel[3];
            size_t index = map.at(y).at(x);
            pixel[0] = colors.at(index)._red;
            pixel[1] = colors.at(index)._green;
            pixel[2] = colors.at(index)._blue;
            out_file.write(reinterpret_cast<char *>(pixel), 3);
        }
        out_file.write(reinterpret_cast<char *>(pad), pad_size);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
save_bmp_file (const std::string name, void* data, DnnInferImageFormatType format, DnnInferPrecisionType precision, uint32_t width, uint32_t height)
{
    std::ofstream out_file;
    out_file.open (name, std::ofstream::binary);
    if (!out_file.is_open ()) {
        return XCAM_RETURN_ERROR_FILE;
    }

    if (height > (size_t)std::numeric_limits<int32_t>::max ||
            width > (size_t)std::numeric_limits<int32_t>::max) {
        XCAM_LOG_ERROR ("File size is too big: %dx%d", height, width);
        return XCAM_RETURN_ERROR_PARAM;
    }

    int32_t pad_size = static_cast<int32_t>(4 - (width * 3) % 4) % 4;
    int32_t size_data = static_cast<int32_t>(width * height * 3 + height * pad_size);
    int32_t size_all = size_data + sizeof(file_header) + sizeof(header_info);

    file_header[2] = (unsigned char)(size_all);
    file_header[3] = (unsigned char)(size_all >> 8);
    file_header[4] = (unsigned char)(size_all >> 16);
    file_header[5] = (unsigned char)(size_all >> 24);

    header_info[4] = (unsigned char)(width);
    header_info[5] = (unsigned char)(width >> 8);
    header_info[6] = (unsigned char)(width >> 16);
    header_info[7] = (unsigned char)(width >> 24);

    int32_t negative_height = -(int32_t)height;
    header_info[8] = (unsigned char)(negative_height);
    header_info[9] = (unsigned char)(negative_height >> 8);
    header_info[10] = (unsigned char)(negative_height >> 16);
    header_info[11] = (unsigned char)(negative_height >> 24);

    header_info[20] = (unsigned char)(size_data);
    header_info[21] = (unsigned char)(size_data >> 8);
    header_info[22] = (unsigned char)(size_data >> 16);
    header_info[23] = (unsigned char)(size_data >> 24);

    out_file.write (reinterpret_cast<char *>(file_header), sizeof(file_header));
    out_file.write (reinterpret_cast<char *>(header_info), sizeof(header_info));

    unsigned char pad[3] = { 0, 0, 0 };
    if (DnnInferPrecisionFP32 == precision) {
        auto data_ptr = reinterpret_cast<float*>(data);
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                unsigned char pixel[3];
                if (DnnInferImageFormatRGBPacked == format) {
                    clamp (data_ptr[y * width * 3 + x * 3], 0.0f, 1.0f);
                    clamp (data_ptr[y * width * 3 + x * 3 + 1], 0.0f, 1.0f);
                    clamp (data_ptr[y * width * 3 + x * 3 + 2], 0.0f, 1.0f);
                    pixel[0] = static_cast<unsigned char>(data_ptr[y * width * 3 + x * 3] * 255);
                    pixel[1] = static_cast<unsigned char>(data_ptr[y * width * 3 + x * 3 + 1] * 255);
                    pixel[2] = static_cast<unsigned char>(data_ptr[y * width * 3 + x * 3 + 2] * 255);
                } else if (DnnInferImageFormatBGRPlanar == format) {
                    clamp (data_ptr[y * width + x + 2 * width * height], 0.0f, 1.0f);
                    clamp (data_ptr[y * width + x + width * height], 0.0f, 1.0f);
                    clamp (data_ptr[y * width + x], 0.0f, 1.0f);
                    pixel[0] = static_cast<unsigned char>(data_ptr[y * width + x + 2 * width * height] * 255);
                    pixel[1] = static_cast<unsigned char>(data_ptr[y * width + x + width * height] * 255);
                    pixel[2] = static_cast<unsigned char>(data_ptr[y * width + x] * 255);
                }
                out_file.write (reinterpret_cast<char *>(pixel), 3);
            }
            out_file.write (reinterpret_cast<char *>(pad), pad_size);
        }
    } else {
        auto data_ptr = reinterpret_cast<unsigned char*>(data);
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                unsigned char pixel[3];
                if (DnnInferImageFormatRGBPacked == format) {
                    pixel[0] = data_ptr[y * width * 3 + x * 3];
                    pixel[1] = data_ptr[y * width * 3 + x * 3 + 1];
                    pixel[2] = data_ptr[y * width * 3 + x * 3 + 2];
                } else if (DnnInferImageFormatBGRPlanar == format) {
                    pixel[0] = data_ptr[y * width + x + 2 * width * height];
                    pixel[1] = data_ptr[y * width + x + width * height];
                    pixel[2] = data_ptr[y * width + x];
                }
                out_file.write (reinterpret_cast<char *>(pixel), 3);
            }
            out_file.write (reinterpret_cast<char *>(pad), pad_size);
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

}  // namespace XCam
