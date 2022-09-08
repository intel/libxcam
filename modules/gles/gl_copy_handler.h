/*
 * gl_copy_handler.h - gl copy handler class
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

#ifndef XCAM_GL_COPY_HANDER_H
#define XCAM_GL_COPY_HANDER_H

#include <xcam_utils.h>
#include <gles/gl_image_shader.h>
#include <gles/gl_image_handler.h>

namespace XCam {

class GLCopyHandler
    : public GLImageHandler
{
public:
    GLCopyHandler (const char *name = "GLCopyHandler");
    ~GLCopyHandler () {}

    bool set_copy_area (uint32_t idx, const Rect &in_area, const Rect &out_area);
    uint32_t get_index () {
        return _index;
    }

    XCamReturn copy (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf);
    virtual XCamReturn terminate ();

protected:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    XCamReturn fix_parameters (const SmartPtr<Parameters> &param);

private:
    uint32_t                   _index;
    Rect                       _in_area;
    Rect                       _out_area;
    SmartPtr<GLImageShader>    _copy_shader;
};

}
#endif // XCAM_GL_COPY_HANDER_H
