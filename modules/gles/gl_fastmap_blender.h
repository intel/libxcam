/*
 * gl_fastmap_blender.h - gl fastmap blender class
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

#ifndef XCAM_GL_FASTMAP_BLENDER_H
#define XCAM_GL_FASTMAP_BLENDER_H

#include <gles/gl_blender.h>
#include <gles/gl_geomap_handler.h>

namespace XCam {

namespace GLFastmapBlendPriv {
class Impl;
}

class GLFastmapBlender
    : public GLImageHandler
{
public:
    explicit GLFastmapBlender (const char *name = "GLFastmapBlender");
    ~GLFastmapBlender ();

    bool set_fastmappers (
        const SmartPtr<GLGeoMapHandler> &left_mapper, const SmartPtr<GLGeoMapHandler> &right_mapper);
    bool set_blender (const SmartPtr<GLBlender> &blender);

    virtual XCamReturn terminate ();

protected:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    SmartPtr<GLGeoMapHandler>             _left_mapper;
    SmartPtr<GLGeoMapHandler>             _right_mapper;
    SmartPtr<GLBlender>                   _blender;

    SmartPtr<GLFastmapBlendPriv::Impl>    _impl;
};

}

#endif // XCAM_GL_FASTMAP_BLENDER_H
