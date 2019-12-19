/*
 * context_cl.h - private context for OpenCL modules
 *
 *  Copyright (c) 2019 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_CONTEXT_CL_H
#define XCAM_CONTEXT_CL_H

#include <string.h>
#include "xcam_utils.h"
#include "context_priv.h"
#include "ocl/cl_image_handler.h"
#include "ocl/cl_context.h"
#include "ocl/cl_blender.h"
#include "interface/stitcher.h"

namespace XCam {

class CLContextBase
    : public ContextBase
{
public:
    virtual ~CLContextBase ();

    virtual XCamReturn init_handler ();
    virtual XCamReturn uinit_handler ();
    virtual bool is_handler_valid () const;

    virtual XCamReturn execute (SmartPtr<VideoBuffer> &buf_in, SmartPtr<VideoBuffer> &buf_out);

protected:
    CLContextBase (HandleType type);

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context) = 0;

private:
    XCAM_DEAD_COPY (CLContextBase);

protected:
    SmartPtr<CLImageHandler>        _handler;
};

class NR3DContext
    : public CLContextBase
{
public:
    NR3DContext ()
        : CLContextBase (HandleType3DNR)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class NRWaveletContext
    : public CLContextBase
{
public:
    NRWaveletContext ()
        : CLContextBase (HandleTypeWaveletNR)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class FisheyeContext
    : public CLContextBase
{
public:
    FisheyeContext ()
        : CLContextBase (HandleTypeFisheye)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class DefogContext
    : public CLContextBase
{
public:
    DefogContext ()
        : CLContextBase (HandleTypeDefog)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class DVSContext
    : public CLContextBase
{
public:
    DVSContext ()
        : CLContextBase (HandleTypeDVS)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);
};

class StitchContext
    : public CLContextBase
{
public:
    StitchContext ()
        : CLContextBase (HandleTypeStitch)
        , _need_seam (false)
        , _fisheye_map (false)
        , _need_lsc (false)
        , _scale_mode (CLBlenderScaleLocal)
        , _res_mode (StitchRes1080P2Cams)
    {}

    virtual SmartPtr<CLImageHandler> create_handler (SmartPtr<CLContext> &context);

private:
    bool                  _need_seam;
    bool                  _fisheye_map;
    bool                  _need_lsc;
    CLBlenderScaleMode    _scale_mode;
    StitchResMode         _res_mode;
};

}

#endif // XCAM_CONTEXT_CL_H