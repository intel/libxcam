/*
 * cv_context.cpp - used to init_opencv_ocl once
 *
 *  Copyright (c) 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Andrey Parfenov <a1994ndrey@gmail.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "cv_context.h"

namespace XCam {

Mutex CVContext::_init_mutex;
SmartPtr<CVContext> CVContext::_instance;

bool CVContext::_ocl_inited = false;

SmartPtr<CVContext>
CVContext::instance ()
{
    SmartLock locker (_init_mutex);
    if (_instance.ptr())
        return _instance;

    _instance = new CVContext ();
    return _instance;
}

bool
CVContext::init_cv_ocl (const char *platform_name, void *platform_id, void *context, void *device_id)
{
    XCAM_ASSERT (!_ocl_inited);
    XCAM_ASSERT (platform_name && platform_id && context && device_id);

    cv::ocl::attachContext (platform_name, platform_id, context, device_id);
    _ocl_inited = cv::ocl::useOpenCL ();
    if(!_ocl_inited) {
        XCAM_LOG_WARNING ("init opencv ocl failed");
        return false;
    }

    return true;
}

void
CVContext::set_ocl (bool use_ocl)
{
    _use_ocl = use_ocl;
}

bool
CVContext::is_ocl_path ()
{
    return _use_ocl;
}

bool
CVContext::is_ocl_inited ()
{
    return _ocl_inited;
}

CVContext::CVContext ()
    : _use_ocl (false)
{
}

CVContext::~CVContext ()
{
}

}
