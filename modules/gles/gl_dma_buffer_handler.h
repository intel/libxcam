/*
 * gl_dma_buffer_handler.h - gl DMA buffer handler class
 *
 *  Copyright (c) 2022 Intel Corporation
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

#ifndef XCAM_GL_DMA_BUFFER_HANDER_H
#define XCAM_GL_DMA_BUFFER_HANDER_H

#include <xcam_utils.h>
#include <gles/gl_image_shader.h>
#include <gles/gl_image_handler.h>
#include <gles/gl_texture.h>

#include <dma_video_buffer.h>

namespace XCam {

namespace GLDmaBufferPriv {
class Impl;
}

enum ShaderID {
    CopyTex2SSBO = 0,
    CopySSBO2Tex,
};

class GLDmaBufferHandler
    : public GLImageHandler
{
    friend class GLDmaBufferPriv::Impl;

public:
    GLDmaBufferHandler (const char *name = "GLDmaBufferHandler");
    ~GLDmaBufferHandler ();

    XCamReturn set_opt_type (const uint32_t type);

    virtual XCamReturn terminate ();

    XCamReturn read_dma_buffer (const SmartPtr<DmaVideoBuffer> &dma_buf, SmartPtr<VideoBuffer> &out_buf);
    XCamReturn write_dma_buffer (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<DmaVideoBuffer> &dma_buf);

protected:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    uint32_t _opt_type;
    SmartPtr<GLDmaBufferPriv::Impl> _impl;
};

}
#endif // XCAM_GL_DMA_BUFFER_HANDER_H
