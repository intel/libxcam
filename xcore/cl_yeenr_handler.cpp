/*
 * cl_yeenr_handler.cpp - CL Y edge enhancement and noise reduction handler
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
#include "xcam_utils.h"
#include "cl_yeenr_handler.h"

namespace XCam {

CLYeenrImageKernel::CLYeenrImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_yeenr")
{
    _yeenr_config.yee_gain = 1.0;
    _yeenr_config.yee_threshold = 1.0;
    _yeenr_config.ynr_gain = 1.0;
}

XCamReturn
CLYeenrImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = output->get_video_info ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    _vertical_offset = video_info.aligned_height;

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_vertical_offset;
    args[2].arg_size = sizeof (_vertical_offset);
    args[3].arg_adress = &_yeenr_config;
    args[3].arg_size = sizeof (CLYeenrConfig);
    arg_count = 4;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info.width;
    work_size.global[1] = video_info.height;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

bool
CLYeenrImageKernel::set_yeenr_ee (const XCam3aResultEdgeEnhancement &ee)
{
    _yeenr_config.yee_gain = ee.gain / 8.0;
    _yeenr_config.yee_threshold = ee.threshold;
    return true;
}

bool
CLYeenrImageKernel::set_yeenr_nr (const XCam3aResultNoiseReduction &nr)
{
    _yeenr_config.ynr_gain = nr.gain;
    return true;
}

CLYeenrImageHandler::CLYeenrImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLYeenrImageHandler::set_yeenr_config_ee (const XCam3aResultEdgeEnhancement &ee)
{
    _yeenr_kernel->set_yeenr_ee(ee);
    return true;
}

bool
CLYeenrImageHandler::set_yeenr_config_nr (const XCam3aResultNoiseReduction &nr)
{
    _yeenr_kernel->set_yeenr_nr(nr);
    return true;
}

bool
CLYeenrImageHandler::set_yeenr_kernel(SmartPtr<CLYeenrImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _yeenr_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_yeenr_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLYeenrImageHandler> yeenr_handler;
    SmartPtr<CLYeenrImageKernel> yeenr_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    yeenr_kernel = new CLYeenrImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_yeenr)
#include "kernel_yeenr.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = yeenr_kernel->load_from_source (kernel_yeenr_body, strlen (kernel_yeenr_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", yeenr_kernel->get_kernel_name());
    }
    XCAM_ASSERT (yeenr_kernel->is_valid ());
    yeenr_handler = new CLYeenrImageHandler ("cl_handler_yeenr");
    yeenr_handler->set_yeenr_kernel (yeenr_kernel);

    return yeenr_handler;
}

}
