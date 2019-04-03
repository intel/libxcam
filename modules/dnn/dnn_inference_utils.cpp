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

XCamReturn
draw_bounding_boxes (
    uint8_t *data,  uint32_t width, uint32_t height,
    std::vector<Vec4i> rectangles, std::vector<int32_t> classes, int32_t thickness)
{
    std::vector<Color> colors = {
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
save_bmp_file (const std::string name, unsigned char* data, uint32_t width, uint32_t height)
{
    std::ofstream out_file;
    out_file.open (name, std::ofstream::binary);
    if (!out_file.is_open ()) {
        return XCAM_RETURN_ERROR_FILE;
    }

    unsigned char file[14] = {
        'B', 'M',           // magic
        0, 0, 0, 0,         // size in bytes
        0, 0,               // app data
        0, 0,               // app data
        40 + 14, 0, 0, 0    // start of data offset
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
        return XCAM_RETURN_ERROR_PARAM;
    }

    int32_t pad_size = static_cast<int32_t>(4 - (width * 3) % 4) % 4;
    int32_t size_data = static_cast<int32_t>(width * height * 3 + height * pad_size);
    int32_t size_all = size_data + sizeof(file) + sizeof(info);

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
            pixel[0] = static_cast<unsigned char>(data[y * width * 3 + x * 3]);
            pixel[1] = static_cast<unsigned char>(data[y * width * 3 + x * 3 + 1]);
            pixel[2] = static_cast<unsigned char>(data[y * width * 3 + x * 3 + 2]);
            out_file.write(reinterpret_cast<char *>(pixel), 3);
        }
        out_file.write(reinterpret_cast<char *>(pad), pad_size);
    }
    return XCAM_RETURN_NO_ERROR;
}

}  // namespace XCam
