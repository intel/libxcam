/*
 * gl_sync.cpp - GL sync implementation
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

#include "gl_sync.h"

namespace XCam {

XCamReturn
GLSync::flush ()
{
    glFlush ();

    GLenum error = gl_error ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GLSync flush failed, error flag: %s", gl_error_string (error));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLSync::finish ()
{
    glFinish ();

    GLenum error = gl_error ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GLSync finish failed, error flag: %s", gl_error_string (error));

    return XCAM_RETURN_NO_ERROR;
}

}
