/*
 * gl_blender.h - gl blender class
 *
 *  Copyright (c) 2018 Intel Corporation
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

#ifndef XCAM_GL_BLENDER_H
#define XCAM_GL_BLENDER_H

#include <interface/blender.h>
#include <gles/gl_buffer.h>
#include <gles/gl_image_handler.h>

#define XCAM_GL_PYRAMID_MAX_LEVEL 4

namespace XCam {

namespace GLBlenderPriv {
class BlenderImpl;
};

class GLBlender
    : public GLImageHandler, public Blender
{
    friend class GLBlenderPriv::BlenderImpl;
    friend SmartPtr<GLImageHandler> create_gl_blender ();

public:
    struct BlenderParam : ImageHandler::Parameters {
        SmartPtr<VideoBuffer> in1_buf;

        BlenderParam (
            const SmartPtr<VideoBuffer> &in0,
            const SmartPtr<VideoBuffer> &in1,
            const SmartPtr<VideoBuffer> &out)
            : Parameters (in0, out)
            , in1_buf (in1)
        {}
    };

public:
    ~GLBlender ();

    bool set_pyr_levels (uint32_t levels);
    const SmartPtr<GLBuffer> &get_layer0_mask () const;

    virtual XCamReturn terminate ();

protected:
    explicit GLBlender (const char *name = "GLBlender");

    XCamReturn blend (
        const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, SmartPtr<VideoBuffer> &out);

    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    SmartPtr<GLBlenderPriv::BlenderImpl>    _impl;
};

extern SmartPtr<GLImageHandler> create_gl_blender ();
}

#endif // XCAM_GL_BLENDER_H
