/*
 * dma_video_buffer.cpp - dma buffer
 *
 *  Copyright (c) 2016 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "dma_video_buffer.h"

namespace XCam {

class DmaVideoBufferPriv
    : public DmaVideoBuffer
{
    friend SmartPtr<DmaVideoBuffer> append_to_dmabuf (XCamVideoBuffer *buf);
protected:
    DmaVideoBufferPriv (const VideoBufferInfo &info, XCamVideoBuffer *buf);
    ~DmaVideoBufferPriv ();

    virtual uint8_t *map ();
    virtual bool unmap ();

private:
    XCamVideoBuffer *_ext_buf;
};

DmaVideoBuffer::DmaVideoBuffer (const VideoBufferInfo &info, int dma_fd, bool need_close_fd)
    : VideoBuffer (info)
    , _dma_fd (dma_fd)
    , _need_close_fd (need_close_fd)
{
    XCAM_ASSERT (dma_fd >= 0);
}

DmaVideoBuffer::~DmaVideoBuffer ()
{
    if (_need_close_fd && _dma_fd > 0)
        close (_dma_fd);
}

uint8_t *
DmaVideoBuffer::map ()
{
    XCAM_ASSERT (false && "DmaVideoBuffer::map failed");
    return NULL;
}

bool
DmaVideoBuffer::unmap ()
{
    XCAM_ASSERT (false && "DmaVideoBuffer::map not supported");
    return false;
}

int
DmaVideoBuffer::get_fd ()
{
    return _dma_fd;
}

DmaVideoBufferPriv::DmaVideoBufferPriv (const VideoBufferInfo &info, XCamVideoBuffer *buf)
    : DmaVideoBuffer (info, buf->get_fd ? xcam_video_buffer_get_fd (buf) : 0, false)
    , _ext_buf (buf)
{
    if (buf->ref)
        xcam_video_buffer_ref (buf);
}

DmaVideoBufferPriv::~DmaVideoBufferPriv ()
{
    if (_ext_buf && _ext_buf->unref && _ext_buf->ref)
        xcam_video_buffer_unref (_ext_buf);
}

uint8_t *
DmaVideoBufferPriv::map ()
{
    uint8_t *mem = _ext_buf->map (_ext_buf);
    XCAM_FAIL_RETURN (ERROR, mem, NULL, "DmaVideoBufferPriv::map failed");

    return mem;
}

bool
DmaVideoBufferPriv::unmap ()
{
    _ext_buf->unmap (_ext_buf);
    return true;
}

SmartPtr<DmaVideoBuffer>
append_to_dmabuf (XCamVideoBuffer *buf)
{
    XCAM_FAIL_RETURN (ERROR, buf, NULL, "append_to_dmabuf failed since buf is NULL");

    if (buf->get_fd) {
        XCAM_FAIL_RETURN (
            ERROR, xcam_video_buffer_get_fd (buf) > 0, NULL,
            "append_to_dmabuf failed, can't get buf file-handle");
    }

    VideoBufferInfo info;
    info.fill (buf->info);

    SmartPtr<DmaVideoBuffer> dmabuf = new DmaVideoBufferPriv (info, buf);
    XCAM_ASSERT (dmabuf.ptr ());

    return dmabuf;
}

}
