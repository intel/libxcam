/*
 * image_file.cpp - Image file implementation
 *
 *  Copyright (c) 2016-2020 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "image_file.h"

namespace XCam {

ImageFile::ImageFile ()
{
}

ImageFile::ImageFile (const char *name, const char *option)
    : File (name, option)
{
}

ImageFile::~ImageFile ()
{
    close ();
}

XCamReturn
ImageFile::read_buf (const SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (is_valid ());

    const VideoBufferInfo &info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;

    uint8_t *memory = buf->map ();
    if (NULL == memory) {
        XCAM_LOG_ERROR ("ImageFile map buffer failed");
        buf->unmap ();
        return XCAM_RETURN_ERROR_MEM;
    }

    for (uint32_t comp = 0; comp < info.components; comp++) {
        info.get_planar_info (planar, comp);
        uint32_t row_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fread (memory + info.offsets [comp] + i * info.strides [comp], 1, row_bytes, _fp) != row_bytes) {
                XCamReturn ret = XCAM_RETURN_NO_ERROR;
                if (end_of_file ()) {
                    ret = XCAM_RETURN_BYPASS;
                } else {
                    XCAM_LOG_ERROR ("ImageFile read file failed, size doesn't match");
                    ret = XCAM_RETURN_ERROR_FILE;
                }

                buf->unmap ();
                return ret;
            }
        }
    }
    buf->unmap ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ImageFile::write_buf (const SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (is_valid ());

    const VideoBufferInfo &info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;

    uint8_t *memory = buf->map ();
    if (NULL == memory) {
        XCAM_LOG_ERROR ("ImageFile map buffer failed");
        buf->unmap ();
        return XCAM_RETURN_ERROR_MEM;
    }

    for (uint32_t comp = 0; comp < info.components; comp++) {
        info.get_planar_info (planar, comp);
        uint32_t row_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fwrite (memory + info.offsets [comp] + i * info.strides [comp], 1, row_bytes, _fp) != row_bytes) {
                XCAM_LOG_ERROR ("ImageFile write file failed, size doesn't match");
                buf->unmap ();
                return XCAM_RETURN_ERROR_FILE;
            }
        }
    }
    buf->unmap ();

    return XCAM_RETURN_NO_ERROR;
}

}
