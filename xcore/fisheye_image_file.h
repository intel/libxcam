/*
 * fisheye_image_file.h - Fisheye image file class
 *
 *  Copyright (c) 2020 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_FISHEYE_IMAGE_FILE_H
#define XCAM_FISHEYE_IMAGE_FILE_H

#include <xcam_std.h>
#include <image_file.h>
#include <video_buffer.h>

namespace XCam {

class FisheyeImageFile
    : public ImageFile
{
public:
    FisheyeImageFile ();
    virtual ~FisheyeImageFile ();

    virtual XCamReturn read_buf (const SmartPtr<VideoBuffer> &buf);

    bool set_fisheye_num (uint32_t num);
    void set_img_size (uint32_t width, uint32_t height);

    void set_center (float cx, float cy, uint32_t idx = 0);
    void set_roi_radius (uint32_t roi_radius, uint32_t idx = 0);

private:
    XCAM_DEAD_COPY (FisheyeImageFile);

    bool gen_roi_pos (uint32_t idx);
    XCamReturn read_roi (const SmartPtr<VideoBuffer> &buf, uint32_t idx);

    enum { FISHEYE_MAX_NUM = 2 };

private:
    uint32_t        _fisheye_num;
    uint32_t        _img_w;
    uint32_t        _img_h;

    float           _cx[FISHEYE_MAX_NUM];
    float           _cy[FISHEYE_MAX_NUM];
    uint32_t        _roi_radius[FISHEYE_MAX_NUM];

    uint32_t        *_x_min[FISHEYE_MAX_NUM];
    uint32_t        *_x_max[FISHEYE_MAX_NUM];

    bool             _update_roi_pos[FISHEYE_MAX_NUM];
};

}

#endif // XCAM_FISHEYE_IMAGE_FILE_H
