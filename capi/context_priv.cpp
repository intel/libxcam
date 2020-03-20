/*
 * context_priv.cpp - capi private context
 *
 *  Copyright (c) 2017 Intel Corporation
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
 */

#include "context_priv.h"
#include "ctxs/context_stitch.h"
#if HAVE_LIBCL
#include "ctxs/context_cl.h"
#endif

using namespace XCam;

#define DEFAULT_INPUT_BUFFER_POOL_COUNT  20

static const char *HandleNames[] = {
    "None",
    "3DNR",
    "WaveletNR",
    "Fisheye",
    "Defog",
    "DVS",
    "Stitch",
    "StitchCL"
};

bool
handle_name_equal (const char *name, HandleType type)
{
    return !strncmp (name, HandleNames[type], strlen(HandleNames[type]));
}

ContextBase::ContextBase (HandleType type)
    : _type (type)
    , _usage (NULL)
    , _image_width (0)
    , _image_height (0)
    , _alloc_out_buf (false)
{
}

ContextBase::~ContextBase ()
{
    xcam_free (_usage);
}

const char*
ContextBase::get_type_name () const
{
    XCAM_ASSERT ((int)_type < sizeof(HandleNames) / sizeof (HandleNames[0]));
    return HandleNames [_type];
}

static const char*
find_value (const ContextParams &param_list, const char *name)
{
    ContextParams::const_iterator i = param_list.find (name);
    if (i != param_list.end ())
        return (i->second);
    return NULL;
}

XCamReturn
ContextBase::set_parameters (ContextParams &param_list)
{
    VideoBufferInfo buf_info;
    uint32_t image_format = V4L2_PIX_FMT_NV12;
    _image_width = 1920;
    _image_height = 1080;

    const char *width = find_value (param_list, "width");
    if (width) {
        _image_width = atoi(width);
    }
    const char *height = find_value (param_list, "height");
    if (height) {
        _image_height = atoi(height);
    }
    if (_image_width == 0 || _image_height == 0) {
        XCAM_LOG_ERROR ("illegal image size width:%d height:%d", _image_width, _image_height);
        return XCAM_RETURN_ERROR_PARAM;
    }

    buf_info.init (image_format, _image_width, _image_height);
    _inbuf_pool->set_video_info (buf_info);
    if (!_inbuf_pool->reserve (DEFAULT_INPUT_BUFFER_POOL_COUNT)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    const char *flag = find_value (param_list, "alloc-out-buf");
    if (flag && !strncasecmp (flag, "true", strlen("true"))) {
        _alloc_out_buf = true;
    } else {
        _alloc_out_buf = false;
    }
    return XCAM_RETURN_NO_ERROR;
}

bool
ContextBase::is_handler_valid () const
{
    XCAM_LOG_ERROR ("handler is invalid in abstract class");
    return false;
}

ContextBase *
create_context (const char *name)
{
    ContextBase *context = NULL;

    if (handle_name_equal (name, HandleTypeNone)) {
        XCAM_LOG_ERROR ("handle type is none");
    } else if (handle_name_equal (name, HandleTypeStitch)) {
        context = new StitchContext;
#if HAVE_LIBCL
    } else if (handle_name_equal (name, HandleType3DNR)) {
        context = new NR3DContext;
    } else if (handle_name_equal (name, HandleTypeWaveletNR)) {
        context = new NRWaveletContext;
    } else if (handle_name_equal (name, HandleTypeFisheye)) {
        context = new FisheyeContext;
    } else if (handle_name_equal (name, HandleTypeDefog)) {
        context = new DefogContext;
    } else if (handle_name_equal (name, HandleTypeDVS)) {
        context = new DVSContext;
    } else if (handle_name_equal (name, HandleTypeStitchCL)) {
        context = new StitchCLContext;
#endif
    } else {
        XCAM_LOG_ERROR ("create context failed with unsupported type:%s", name);
        return NULL;
    }

    return context;
}

