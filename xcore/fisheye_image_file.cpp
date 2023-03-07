/*
 * fisheye_image_file.cpp - Fisheye image file implementation
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

#include "fisheye_image_file.h"

namespace XCam {

FisheyeImageFile::FisheyeImageFile ()
    : _fisheye_num (1)
    , _img_w (0)
    , _img_h (0)
{
    xcam_mem_clear (_cx);
    xcam_mem_clear (_cy);
    xcam_mem_clear (_roi_radius);

    xcam_mem_clear (_x_min);
    xcam_mem_clear (_x_max);

    for (uint32_t idx = 0; idx < FISHEYE_MAX_NUM; ++idx) {
        _update_roi_pos[idx] = true;
    }
}

FisheyeImageFile::~FisheyeImageFile ()
{
    for (uint32_t idx = 0; idx < FISHEYE_MAX_NUM; ++idx) {
        if (_x_min[idx])
            xcam_free (_x_min[idx]);
        _x_min[idx] = NULL;

        if (_x_max[idx])
            xcam_free (_x_max[idx]);
        _x_max[idx] = NULL;
    }
}

bool
FisheyeImageFile::set_fisheye_num (uint32_t num)
{
    XCAM_FAIL_RETURN (
        ERROR, num <= FISHEYE_MAX_NUM, false,
        "FisheyeImageFile fisheye number(%d) should not be greater than %d",
        num, FISHEYE_MAX_NUM);

    _fisheye_num = num;

    return true;
}

void
FisheyeImageFile::set_img_size (uint32_t width, uint32_t height)
{
    _img_w = width;
    _img_h = height;
}

void
FisheyeImageFile::set_center (float cx, float cy, uint32_t idx)
{
    _cx[idx] = cx;
    _cy[idx] = cy;
}

void
FisheyeImageFile::set_roi_radius (uint32_t roi_radius, uint32_t idx)
{
    if (roi_radius != _roi_radius[idx])
        _update_roi_pos[idx] = true;

    _roi_radius[idx] = roi_radius;
}

bool
FisheyeImageFile::gen_roi_pos (uint32_t idx)
{
    if (!_x_min[idx])
        _x_min[idx] = (uint32_t *) xcam_malloc0 (_img_h * sizeof (uint32_t));
    if (!_x_max[idx])
        _x_max[idx] = (uint32_t *) xcam_malloc0 (_img_h * sizeof (uint32_t));

    float y_sqrt;
    for (uint32_t y = 0; y < _img_h; ++y) {
        y_sqrt = sqrt (_roi_radius[idx] * _roi_radius[idx] - (y - _cy[idx] ) * (y - _cy[idx] ));

        _x_min[idx][y] = (_cx[idx] > y_sqrt) ? (_cx[idx] - y_sqrt) : 0.0f;
        _x_max[idx][y] = _cx[idx] + y_sqrt + 1.5f;
        _x_max[idx][y] = (_x_max[idx][y] > _img_w) ? _img_w : _x_max[idx][y];
    }

    return true;
}

XCamReturn
FisheyeImageFile::read_roi (const SmartPtr<VideoBuffer> &buf, uint32_t idx)
{
    if (_update_roi_pos[idx]) {
        gen_roi_pos (idx);
        _update_roi_pos[idx] = false;
    }

    const VideoBufferInfo &info = buf->get_video_info ();

    uint8_t *memory = buf->map ();
    if (NULL == memory) {
        XCAM_LOG_ERROR ("FisheyeImageFile map buffer failed");
        buf->unmap ();
        return XCAM_RETURN_ERROR_MEM;
    }

    for (uint32_t comp = 0; comp < info.components; comp++) {
        VideoBufferPlanarInfo planar;
        info.get_planar_info (planar, comp);

        uint32_t x_step = info.width / planar.width;
        uint32_t y_step = info.height / planar.height;
        uint8_t *start = memory + info.offsets [comp];

        uint32_t x_min, x_max, bytes;
        uint32_t h = 0;
        uint32_t fp_offset = 0;

        for (uint32_t i = 0; i < planar.height; i++) {
            x_min = _x_min[idx][h] / x_step;
            x_max = (_x_max[idx][h] + x_step - 1) / x_step;

            if (fseek (_fp, fp_offset + x_min, SEEK_CUR) < 0)
                return XCAM_RETURN_ERROR_MEM;

	    bytes = (x_max - x_min) * planar.pixel_bytes;

            if (fread (start + x_min, 1, bytes, _fp) != bytes) {
                XCamReturn ret = XCAM_RETURN_NO_ERROR;
                if (end_of_file ()) {
                    ret = XCAM_RETURN_BYPASS;
                } else {
                    XCAM_LOG_ERROR ("FisheyeImageFile read file failed, size doesn't match");
                    ret = XCAM_RETURN_ERROR_FILE;
                }

                buf->unmap ();
                return ret;
            }

            fp_offset = info.strides [comp] - x_max;
            start += info.strides [comp];
            h += y_step;
        }

        if (fseek (_fp, fp_offset, SEEK_CUR) < 0)
            return XCAM_RETURN_ERROR_MEM;

    }
    buf->unmap ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
FisheyeImageFile::read_buf (const SmartPtr<VideoBuffer> &buf)
{
    fpos_t cur_pos;
    fgetpos (_fp, &cur_pos);

    for (uint32_t idx = 0; idx < _fisheye_num; ++idx) {
        fsetpos(_fp, &cur_pos);

        XCamReturn ret = read_roi (buf, idx);
        if (ret != XCAM_RETURN_NO_ERROR)
            return ret;
    }

    return XCAM_RETURN_NO_ERROR;
}

}
