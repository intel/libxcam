/*
 * external_video_buffer_priv.cpp
 *
 *  Copyright (c) 2017 Intel Corporation
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

#include <xcam_std.h>
#include <video_buffer.h>

namespace XCam {

class XCamExternalVideoBuffer
    : public VideoBuffer
{
public:
    XCamExternalVideoBuffer (const VideoBufferInfo &info, uint8_t* buffer);

    virtual ~XCamExternalVideoBuffer ();

    virtual uint8_t *map ();
    virtual bool unmap ();
    virtual int get_fd ();

private:

    XCAM_DEAD_COPY (XCamExternalVideoBuffer);

private:
    uint8_t* _buffer;
};

XCamExternalVideoBuffer::XCamExternalVideoBuffer (const VideoBufferInfo &info, uint8_t* buffer)
    : VideoBuffer (info)
    , _buffer (buffer)
{
    XCAM_ASSERT (buffer != NULL);
}

XCamExternalVideoBuffer::~XCamExternalVideoBuffer ()
{
}

uint8_t *
XCamExternalVideoBuffer::map ()
{
    return _buffer;
}

bool
XCamExternalVideoBuffer::unmap ()
{
    return true;
}

int
XCamExternalVideoBuffer::get_fd ()
{
    XCAM_ASSERT (false && "XCamExternalVideoBuffer::get_fd not supported");
    return -1;
}

SmartPtr<VideoBuffer>
external_buf_to_xcam_video_buf (
    uint8_t* buf, uint32_t format,
    uint32_t width, uint32_t height,
    uint32_t aligned_width, uint32_t aligned_height,
    uint32_t size)
{
    VideoBufferInfo buf_info;
    SmartPtr<XCamExternalVideoBuffer> video_buffer;

    XCAM_FAIL_RETURN (
        ERROR, buf, NULL,
        "external_buf_to_xcam_video_buf failed since buf is NULL");

    buf_info.init (format, width, height,
                   aligned_width, aligned_height, size);
    video_buffer = new XCamExternalVideoBuffer (buf_info, buf);
    XCAM_ASSERT (video_buffer.ptr ());
    return video_buffer;
}

}
