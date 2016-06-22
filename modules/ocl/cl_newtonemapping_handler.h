/*
 * cl_newtonemapping_handler.h - CL tonemapping handler
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
 * Author: Wu Junkai <junkai.wu@intel.com>
 */

#ifndef XCAM_CL_NEWTONEMAPPING_HANLDER_H
#define XCAM_CL_NEWTONEMAPPING_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "x3a_stats_pool.h"


namespace XCam {

class CLNewTonemappingImageKernel
    : public CLImageKernel
{
public:
    explicit CLNewTonemappingImageKernel (SmartPtr<CLContext> &context,
                                          const char *name);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLNewTonemappingImageKernel);
    int                     _image_width;
    int                     _image_height;
    int                     _block_factor;
    float                   _map_hist[65536];
    float                   _y_max[16];
    float                   _y_avg[16];
    SmartPtr<CLBuffer>      _y_max_buffer;
    SmartPtr<CLBuffer>      _y_avg_buffer;
    SmartPtr<CLBuffer>      _map_hist_buffer;
};

class CLNewTonemappingImageHandler
    : public CLImageHandler
{
public:
    explicit CLNewTonemappingImageHandler (const char *name);
    bool set_tonemapping_kernel(SmartPtr<CLNewTonemappingImageKernel> &kernel);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCAM_DEAD_COPY (CLNewTonemappingImageHandler);
    SmartPtr<CLNewTonemappingImageKernel>  _tonemapping_kernel;
    int32_t  _output_format;
};

SmartPtr<CLImageHandler>
create_cl_newtonemapping_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_NEWTONEMAPPING_HANLDER_H
