/*
 * cl_denoise_handler.cpp - CL denoise handler
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: Juan Zhao <juan.j.zhao@intel.com>
 */



#include "xcam_utils.h"
#include "cl_denoise_handler.h"

namespace XCam {

CLDenoiseImageKernel::CLDenoiseImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_denoise")
{
}

XCamReturn
CLDenoiseImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();
    cl_libva_image image_info;
    uint32_t channel_bits = XCAM_ALIGN_UP (video_info.color_bits, 8);

    xcam_mem_clear (&image_info);
    image_info.fmt.image_channel_order = CL_RGBA;
    if (channel_bits == 8)
        image_info.fmt.image_channel_data_type = CL_UNORM_INT8;
    else if (channel_bits == 16)
        image_info.fmt.image_channel_data_type = CL_UNORM_INT16;
    image_info.offset = 0;
    image_info.width = video_info.width;
    image_info.height = (video_info.size / video_info.strides[0])/4*4;
    image_info.row_pitch = video_info.strides[0];
    imw = video_info.width;
    imh = video_info.height;
    //sigma_r = 0.1*100
    sigma_r = 10;

    _image_in = new CLVaImage (context, input, &image_info);
    _image_out = new CLVaImage (context, output, &image_info);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &sigma_r;
    args[2].arg_size = sizeof (sigma_r);
    args[3].arg_adress = &imw;
    args[3].arg_size = sizeof (imw);
    args[4].arg_adress = &imh;
    args[4].arg_size = sizeof (imh);
    arg_count = 5;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = imh;
    work_size.global[1] = imw/4;
    work_size.local[0] = imh/270;
    work_size.local[1] = imw/64;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLDenoiseImageKernel::post_execute ()
{
    return CLImageKernel::post_execute ();
}

SmartPtr<CLImageHandler>
create_cl_denoise_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> denoise_handler;
    SmartPtr<CLImageKernel> denoise_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    denoise_kernel = new CLDenoiseImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_denoise)
#include "kernel_denoise.cl"
        XCAM_CL_KERNEL_FUNC_END;
        ret = denoise_kernel->load_from_source (kernel_denoise_body, strlen (kernel_denoise_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", denoise_kernel->get_kernel_name());
    }
    XCAM_ASSERT (denoise_kernel->is_valid ());
    denoise_handler = new CLImageHandler ("cl_handler_denoise");
    denoise_handler->add_kernel  (denoise_kernel);

    return denoise_handler;
}

};
