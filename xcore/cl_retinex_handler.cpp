/*
 * cl_retinex_handler.cpp - CL retinex handler
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
#include "cl_retinex_handler.h"
#include <algorithm>

namespace XCam {

CLRetinexImageKernel::CLRetinexImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_retinex")
    , _vertical_offset_in (0)
    , _vertical_offset_out (0)
{
    set_gaussian(5, 2);
}

bool
CLRetinexImageKernel::set_gaussian (int size, float sigma)
{
    int i, j;
    float dis=0, sum=0;
    for(i=0; i<size; i++)  {
	for(j=0; j<size; j++)  {
			{
                                dis = (float)(i - size/2) * (i - size/2) + (j -size/2) * (j - size/2);
                                _g_table[i * XCAM_RETINEX_TABLE_SIZE + j] = 1/(2 * 3.14 * sigma * sigma) * exp(-dis/(2 * sigma * sigma));
                                sum += _g_table[i * XCAM_RETINEX_TABLE_SIZE + j];
		    }
		}
	}

    for(i=0; i<XCAM_RETINEX_TABLE_SIZE*XCAM_RETINEX_TABLE_SIZE; i++)
	_g_table[i] = _g_table[i] / sum;

    return true;
}


XCamReturn
CLRetinexImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info_in = input->get_video_info ();
    const VideoBufferInfo & video_info_out = output->get_video_info ();
    _retinex_config.log_min = -0.192321;
    _retinex_config.log_max = 0.122745;
    _retinex_config.gain = 255.0/(_retinex_config.log_max - _retinex_config.log_min);

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);
    _g_table_buffer = new CLBuffer(
        context, sizeof(float)*XCAM_RETINEX_TABLE_SIZE*XCAM_RETINEX_TABLE_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_g_table);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    _vertical_offset_in = video_info_in.aligned_height;
    _vertical_offset_out= video_info_out.aligned_height;

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    args[2].arg_adress = &_vertical_offset_in;
    args[2].arg_size = sizeof (_vertical_offset_in);
    args[3].arg_adress = &_vertical_offset_out;
    args[3].arg_size = sizeof (_vertical_offset_out);
    args[4].arg_adress = &_retinex_config;
    args[4].arg_size = sizeof (CLRetinexConfig);
    args[5].arg_adress = &_g_table_buffer->get_mem_id();
    args[5].arg_size = sizeof (cl_mem);
    arg_count = 6;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info_in.width;
    work_size.global[1] = video_info_in.height;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

CLRetinexImageHandler::CLRetinexImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLRetinexImageHandler::set_gaussian_table (int size, float sigma)
{
    _retinex_kernel->set_gaussian (size, sigma);
    return true;
}

bool
CLRetinexImageHandler::set_retinex_kernel(SmartPtr<CLRetinexImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _retinex_kernel = kernel;
    return true;
}

SmartPtr<CLImageHandler>
create_cl_retinex_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLRetinexImageHandler> retinex_handler;
    SmartPtr<CLRetinexImageKernel> retinex_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    retinex_kernel = new CLRetinexImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_retinex)
#include "kernel_retinex.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = retinex_kernel->load_from_source (kernel_retinex_body, strlen (kernel_retinex_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", retinex_kernel->get_kernel_name());
    }
    XCAM_ASSERT (retinex_kernel->is_valid ());
    retinex_handler = new CLRetinexImageHandler ("cl_handler_retinex");
    retinex_handler->set_retinex_kernel (retinex_kernel);

    return retinex_handler;
}

}
