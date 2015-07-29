/*
 * cl_yuv_pipe_handler.cpp - CL YuvPipe Pipe handler
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
 * Author: Wangfei <feix.w.wang@intel.com>
 */
#include "xcam_utils.h"
#include "cl_yuv_pipe_handler.h"
float default_matrix[XCAM_COLOR_MATRIX_SIZE] = {0.299, 0.587, 0.114, -0.14713, -0.28886, 0.436, 0.615, -0.51499, -0.10001};
float default_macc[XCAM_CHROMA_AXIS_SIZE*XCAM_CHROMA_MATRIX_SIZE] = {
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000,
    1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000
};

namespace XCam {

CLYuvPipeImageKernel::CLYuvPipeImageKernel (SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_yuv_pipe")
    , _vertical_offset (0)
{
    set_macc (default_macc);
    set_matrix (default_matrix);
}

bool
CLYuvPipeImageKernel::set_macc (float *macc)
{
    memcpy(_macc_table, macc, sizeof(float)*XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE);
    return true;
}

bool
CLYuvPipeImageKernel::set_matrix (float *matrix)
{
    memcpy(_rgbtoyuv_matrix, matrix, sizeof(float)*XCAM_COLOR_MATRIX_SIZE);
    return true;
}

XCamReturn
CLYuvPipeImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();


    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);
    _matrix_buffer = new CLBuffer (
        context, sizeof(float)*XCAM_COLOR_MATRIX_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_rgbtoyuv_matrix);
    _macc_table_buffer = new CLBuffer(
        context, sizeof(float)*XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_macc_table);


    _vertical_offset = video_info.aligned_height;

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
    args[2].arg_adress = &_vertical_offset;
    args[2].arg_size = sizeof (_vertical_offset);
    args[3].arg_adress = &_matrix_buffer->get_mem_id();
    args[3].arg_size = sizeof (cl_mem);
    args[4].arg_adress = &_macc_table_buffer->get_mem_id();
    args[4].arg_size = sizeof (cl_mem);

    arg_count = 5;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = video_info.width / 2 ;
    work_size.global[1] = video_info.height / 2 ;
    work_size.local[0] = 4;
    work_size.local[1] = 4;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLYuvPipeImageKernel::post_execute ()
{
    _image_in.release ();
    _image_out.release ();
    _matrix_buffer.release ();
    _macc_table_buffer.release ();

    return XCAM_RETURN_NO_ERROR;
}

CLYuvPipeImageHandler::CLYuvPipeImageHandler (const char *name)
    : CLImageHandler (name)
    , _output_format(V4L2_PIX_FMT_NV12)
{
}

bool
CLYuvPipeImageHandler::set_macc_table (const XCam3aResultMaccMatrix &macc)
{
    float macc_table[XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE];
    for(int i = 0; i < XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE; i++)
        macc_table[i] = (float)macc.table[i];
    _yuv_pipe_kernel->set_macc(macc_table);
    return true;
}

bool
CLYuvPipeImageHandler::set_rgbtoyuv_matrix (const XCam3aResultColorMatrix &matrix)
{
    float matrix_table[XCAM_COLOR_MATRIX_SIZE];
    for (int i = 0; i < XCAM_COLOR_MATRIX_SIZE; i++)
        matrix_table[i] = (float)matrix.matrix[i];
    _yuv_pipe_kernel->set_matrix(matrix_table);
    return true;
}

XCamReturn
CLYuvPipeImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    bool format_inited = output.init (_output_format, input.width, input.height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) ouput format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (_output_format));

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_yuv_pipe_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> yuv_pipe_handler;
    SmartPtr<CLImageKernel> yuv_pipe_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    yuv_pipe_kernel = new CLYuvPipeImageKernel (context);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_yuv_pipe)
#include "kernel_yuv_pipe.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = yuv_pipe_kernel->load_from_source (kernel_yuv_pipe_body, strlen (kernel_yuv_pipe_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", yuv_pipe_kernel->get_kernel_name());
    }
    XCAM_ASSERT (yuv_pipe_kernel->is_valid ());
    yuv_pipe_handler = new CLYuvPipeImageHandler ("cl_handler_pipe_yuv");
    yuv_pipe_handler->add_kernel  (yuv_pipe_kernel);

    return yuv_pipe_handler;
}

};
