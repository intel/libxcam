/*
 * cl_yeenr_handler.h - CL Y edge enhancement and noise reduction handler
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: wangfei <feix.w.wang@intel.com>
 */

#ifndef XCAM_CL_YEENR_HANLDER_H
#define XCAM_CL_YEENR_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

typedef struct {
    float           yee_gain;
    float           yee_threshold;
    float           ynr_gain;
} CLYeenrConfig;

class CLYeenrImageKernel
    : public CLImageKernel
{
public:
    explicit CLYeenrImageKernel (SmartPtr<CLContext> &context);
    bool set_yeenr_ee (const XCam3aResultEdgeEnhancement &ee);
    bool set_yeenr_nr (const XCam3aResultNoiseReduction &nr);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLYeenrImageKernel);
    uint32_t _vertical_offset;
    CLYeenrConfig _yeenr_config;
};

class CLYeenrImageHandler
    : public CLImageHandler
{
public:
    explicit CLYeenrImageHandler (const char *name);
    bool set_yeenr_config_ee (const XCam3aResultEdgeEnhancement &ee);
    bool set_yeenr_config_nr (const XCam3aResultNoiseReduction &nr);
    bool set_yeenr_kernel(SmartPtr<CLYeenrImageKernel> &kernel);

private:
    XCAM_DEAD_COPY (CLYeenrImageHandler);
    SmartPtr<CLYeenrImageKernel> _yeenr_kernel;
};

SmartPtr<CLImageHandler>
create_cl_yeenr_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_YEENR_HANLDER_H
