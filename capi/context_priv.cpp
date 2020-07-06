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
    "none",
    "3dnr",
    "waveletnr",
    "fisheye",
    "defog",
    "dvs",
    "stitch",
    "stitchcl"
};

bool
handle_name_equal (const char *name, HandleType type)
{
    return !strcmp (name, HandleNames[type]);
}

ContextBase::ContextBase (HandleType type)
    : _type (type)
    , _usage (NULL)
    , _input_width (0)
    , _input_height (0)
    , _output_width (0)
    , _output_height (0)
    , _format (V4L2_PIX_FMT_NV12)
    , _mem_type (XCAM_MEM_TYPE_CPU)
    , _alloc_out_buf (0)
{
}

ContextBase::~ContextBase ()
{
    xcam_free (_usage);
}

void
ContextBase::parse_value (const ContextParams &params, const char *name, uint32_t &value)
{
    ContextParams::const_iterator iter = params.find (name);
    if (iter == params.end ())
        return;
    value = atoi (iter->second);
}

const char *
ContextBase::get_type_name () const
{
    XCAM_ASSERT ((int)_type < sizeof(HandleNames) / sizeof (HandleNames[0]));
    return HandleNames [_type];
}

XCamReturn
ContextBase::set_parameters (ContextParams &param_list)
{
    parse_value (param_list, "fmt", _format);
    parse_value (param_list, "allocoutbuf", _alloc_out_buf);

    parse_value (param_list, "inw", _input_width);
    parse_value (param_list, "inh", _input_height);
    XCAM_FAIL_RETURN (
        ERROR, _input_width || _input_height , XCAM_RETURN_ERROR_PARAM,
        "illegal input size %dx%d", _input_width, _input_height);

    VideoBufferInfo info;
    info.init (_format, _input_width, _input_height);
    _inbuf_pool->set_video_info (info);
    XCAM_FAIL_RETURN (
        ERROR, _inbuf_pool->reserve (DEFAULT_INPUT_BUFFER_POOL_COUNT), XCAM_RETURN_ERROR_PARAM,
        "init input buffer pool failed");

    parse_value (param_list, "outw", _output_width);
    parse_value (param_list, "outh", _output_height);
    XCAM_FAIL_RETURN (
        ERROR, _output_width || _output_height , XCAM_RETURN_ERROR_PARAM,
        "illegal output size %dx%d", _output_width, _output_height);

    return XCAM_RETURN_NO_ERROR;
}

bool
ContextBase::is_handler_valid () const
{
    XCAM_LOG_ERROR ("handler is invalid in abstract class");
    return false;
}

void
ContextBase::set_buf_pool (const SmartPtr<BufferPool> &pool)
{
    _inbuf_pool = pool;
}

void
ContextBase::set_mem_type (uint32_t type)
{
    _mem_type = type;
}

uint32_t
ContextBase::get_mem_type () const
{
    return _mem_type;
}

bool
ContextBase::need_alloc_out_buf () const
{
    return _alloc_out_buf;
}

uint32_t
ContextBase::get_in_width () const
{
    return _input_width;
}

uint32_t
ContextBase::get_in_height () const
{
    return _input_height;
}

uint32_t
ContextBase::get_out_width () const
{
    return _output_width;
}

uint32_t
ContextBase::get_out_height () const
{
    return _output_height;
}

uint32_t
ContextBase::get_format () const
{
    return _format;
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

