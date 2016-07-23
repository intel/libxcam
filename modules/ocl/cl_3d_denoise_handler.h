/*
 * cl_3d_denoise_handler.h - CL 3D noise reduction handler
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
 * Author: Wei Zong <wei.zong@intel.com>
 */

#ifndef XCAM_CL_3D_DENOISE_HANLDER_H
#define XCAM_CL_3D_DENOISE_HANLDER_H

#include "xcam_utils.h"
#include "base/xcam_3a_result.h"
#include "x3a_stats_pool.h"
#include "cl_image_handler.h"

namespace XCam {

class CL3DDenoiseImageHandler;

class CL3DDenoiseImageKernel
    : public CLImageKernel
{
    typedef std::list<SmartPtr<CLImage>> CLImagePtrList;

private:

public:
    explicit CL3DDenoiseImageKernel (SmartPtr<CLContext> &context,
                                     const char *name,
                                     uint32_t channel,
                                     SmartPtr<CL3DDenoiseImageHandler> &handler);

    virtual ~CL3DDenoiseImageKernel () {
        _image_in_list.clear ();
    }

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CL3DDenoiseImageKernel);

    uint32_t _channel;
    float    _gain;
    float    _thr_y;
    float    _thr_uv;
    uint8_t  _ref_count;
    SmartPtr<CL3DDenoiseImageHandler> _handler;

    CLImagePtrList _image_in_list;
};

class CL3DDenoiseImageHandler
    : public CLImageHandler
{
public:
    explicit CL3DDenoiseImageHandler (const char *name);

    bool set_ref_framecount (const uint8_t count);
    uint8_t get_ref_framecount () const {
        return _ref_count;
    };

    bool set_denoise_config (const XCam3aResultTemporalNoiseReduction& config);
    XCam3aResultTemporalNoiseReduction& get_denoise_config () {
        return _config;
    };

private:
    XCAM_DEAD_COPY (CL3DDenoiseImageHandler);

private:
    uint8_t _ref_count;
    XCam3aResultTemporalNoiseReduction _config;
};

SmartPtr<CLImageHandler>
create_cl_3d_denoise_image_handler (SmartPtr<CLContext> &context, uint32_t channel);

};

#endif //XCAM_CL_3D_DENOISE_HANLDER_H
