/*
 * image_file.h - Image file class
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

#ifndef XCAM_IMAGE_FILE_H
#define XCAM_IMAGE_FILE_H

#include <xcam_std.h>
#include <file.h>
#include <video_buffer.h>

namespace XCam {

class ImageFile
    : public File
{
public:
    ImageFile ();
    explicit ImageFile (const char *name, const char *option);
    virtual ~ImageFile ();

    virtual XCamReturn read_buf (const SmartPtr<VideoBuffer> &buf);
    XCamReturn write_buf (const SmartPtr<VideoBuffer> &buf);

private:
    XCAM_DEAD_COPY (ImageFile);
};

}

#endif // XCAM_IMAGE_FILE_H
