/*
 * context_cl.cpp - private context for OpenCL modules
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

#include "context_cl.h"
#include "ocl/cl_device.h"
#include "ocl/cl_utils.h"
#include "ocl/cl_newwavelet_denoise_handler.h"
#include "ocl/cl_defog_dcp_handler.h"
#include "ocl/cl_3d_denoise_handler.h"
#include "ocl/cl_image_warp_handler.h"
#include "ocl/cl_fisheye_handler.h"
#include "ocl/cl_image_360_stitch.h"

namespace XCam {

CLContextBase::CLContextBase (HandleType type)
    : ContextBase (type)
{
    if (!get_input_buffer_pool ().ptr()) {
        SmartPtr<BufferPool> pool = new CLVideoBufferPool ();
        XCAM_ASSERT (pool.ptr ());

        set_buf_pool (pool);
    }
}

CLContextBase::~CLContextBase ()
{
}

XCamReturn
CLContextBase::init_handler ()
{
    SmartPtr<CLContext> cl_context = CLDevice::instance()->get_context ();
    XCAM_FAIL_RETURN (
        ERROR, cl_context.ptr (), XCAM_RETURN_ERROR_UNKNOWN,
        "CLContextBase::init_handler(%s) failed since cl-context is NULL",
        get_type_name ());

    SmartPtr<CLImageHandler> handler = create_handler (cl_context);
    XCAM_FAIL_RETURN (
        ERROR, handler.ptr (), XCAM_RETURN_ERROR_UNKNOWN,
        "CLContextBase::init_handler(%s) create handler failed", get_type_name ());

    handler->disable_buf_pool (!need_alloc_out_buf ());
    _handler = handler;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLContextBase::uinit_handler ()
{
    if (!_handler.ptr ())
        return XCAM_RETURN_NO_ERROR;

    _handler->emit_stop ();
    _handler.release ();
    return XCAM_RETURN_NO_ERROR;
}

bool
CLContextBase::is_handler_valid () const
{
    return _handler.ptr () ? true : false;
}

XCamReturn
CLContextBase::execute (SmartPtr<VideoBuffer> &buf_in, SmartPtr<VideoBuffer> &buf_out)
{
    if (!need_alloc_out_buf ()) {
        XCAM_FAIL_RETURN (
            ERROR, buf_out.ptr (), XCAM_RETURN_ERROR_MEM,
            "context (%s) execute failed, buf_out need set.", get_type_name ());
    } else {
        XCAM_FAIL_RETURN (
            ERROR, !buf_out.ptr (), XCAM_RETURN_ERROR_MEM,
            "context (%s) execute failed, buf_out need NULL.", get_type_name ());
    }

    return _handler->execute (buf_in, buf_out);
}

SmartPtr<CLImageHandler>
NR3DContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_3d_denoise_image_handler (
               context, CL_IMAGE_CHANNEL_Y | CL_IMAGE_CHANNEL_UV, 3);
}

SmartPtr<CLImageHandler>
NRWaveletContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_newwavelet_denoise_image_handler (
               context, CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y, false);
}

SmartPtr<CLImageHandler>
FisheyeContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_fisheye_handler (context);
}

SmartPtr<CLImageHandler>
DefogContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_defog_dcp_image_handler (context);;
}

SmartPtr<CLImageHandler>
DVSContext::create_handler (SmartPtr<CLContext> &context)
{
    return create_cl_image_warp_handler (context);
}

SmartPtr<CLImageHandler>
StitchContext::create_handler (SmartPtr<CLContext> &context)
{
    uint32_t sttch_width = get_image_width ();
    uint32_t sttch_height = XCAM_ALIGN_UP (sttch_width / 2, 16);
    if (sttch_width != sttch_height * 2) {
        XCAM_LOG_ERROR ("incorrect stitch size width:%d height:%d", sttch_width, sttch_height);
        return NULL;
    }

    FisheyeDewarpMode dewarp_mode = DewarpSphere;
    StitchResMode res_mode = (_res_mode == StitchRes1080P2Cams) ? StitchRes4K2Cams : StitchRes4K2Cams;

    SmartPtr<CLImage360Stitch> image_360 =
        create_image_360_stitch (context, _need_seam, _scale_mode,
                                 _fisheye_map, _need_lsc, dewarp_mode, res_mode).dynamic_cast_ptr<CLImage360Stitch> ();
    XCAM_FAIL_RETURN (ERROR, image_360.ptr (), NULL, "create image stitch handler failed");
    image_360->set_output_size (sttch_width, sttch_height);
    XCAM_LOG_INFO ("stitch output size width:%d height:%d", sttch_width, sttch_height);

    return image_360;
}

}

