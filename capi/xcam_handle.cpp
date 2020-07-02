/*
 * xcam_handle.cpp - xcam handle implementation
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

#include "xcam_utils.h"
#include "xcam_handle.h"
#include "dma_video_buffer.h"
#include "context_priv.h"
#include <stdarg.h>

using namespace XCam;

#define CONTEXT_BASE_CAST(handle) (ContextBase*)(handle)
#define HANDLE_CAST(context) (XCamHandle*)(context)

XCamHandle *
xcam_create_handle (const char *name)
{
    ContextBase *context = create_context (name);
    return HANDLE_CAST (context);
}

void
xcam_destroy_handle (XCamHandle *handle)
{
    if (handle)
        delete CONTEXT_BASE_CAST (handle);
}

XCamReturn
xcam_handle_init (XCamHandle *handle)
{
    ContextBase *context = CONTEXT_BASE_CAST (handle);
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, context, XCAM_RETURN_ERROR_PARAM,
        "xcam_handler_init failed, handle can NOT be NULL, did you have xcam_create_handle first?");

    ret = context->init_handler ();
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "xcam_handler_init, create handle ptr(%s) failed", context->get_type_name ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
xcam_handle_uinit (XCamHandle *handle)
{
    ContextBase *context = CONTEXT_BASE_CAST (handle);

    XCAM_FAIL_RETURN (
        ERROR, context, XCAM_RETURN_ERROR_PARAM,
        "xcam_handler_uinit failed, handle can NOT be NULL");

    return context->uinit_handler ();
}

XCamReturn
xcam_handle_get_usage (XCamHandle *handle, char *usage_buf, int *usage_len)
{
    ContextBase *context = CONTEXT_BASE_CAST (handle);
    XCAM_FAIL_RETURN (
        ERROR, context, XCAM_RETURN_ERROR_PARAM,
        "xcam_handle_get_usage failed, handle can NOT be NULL");

    const char *usage = context->get_usage ();
    int len = strlen (usage) + 1;
    if (len < *usage_len)
        len = *usage_len;
    strncpy (usage_buf, usage, len - 1);
    *usage_len = len;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
xcam_handle_set_parameters (XCamHandle *handle, const char *params)
{
    ContextBase *context = CONTEXT_BASE_CAST (handle);
    XCAM_FAIL_RETURN (
        ERROR, context, XCAM_RETURN_ERROR_PARAM,
        "xcam_handle_set_parameters failed, handle can NOT be NULL");

    ContextParams ctx_params;
    char pairs[XCAM_MAX_PARAMS_LENGTH] = { 0 };
    strncpy (pairs, params, XCAM_MAX_PARAMS_LENGTH - 1);

    char *pair = pairs;
    char *saveptr = NULL;
    char *value = NULL;
    while ((pair = strtok_r (pair, " ", &saveptr)) != NULL)
    {
        char *field = strtok_r (pair, "=", &value);
        XCAM_FAIL_RETURN (
            ERROR, value, XCAM_RETURN_ERROR_PARAM,
            "xcam_handle(%s) set parameters failed, param(%s) should never be NULL",
            context->get_type_name (), field);

        ctx_params[field] = value;
        pair = NULL;
    }

    return context->set_parameters (ctx_params);
}

#if 0
XCamReturn
xcam_handle_set_parameters (
    XCamHandle *handle, const char *field, ...)
{
    ContextBase *context = CONTEXT_BASE_CAST (handle);
    ContextParams params;

    XCAM_FAIL_RETURN (
        ERROR, context, XCAM_RETURN_ERROR_PARAM,
        "xcam_handle_set_parameters failed, handle can NOT be NULL");

    const char *vfield, *vvalue;
    vfield = field;
    va_list args;
    va_start (args, field);
    while (vfield) {
        vvalue = va_arg (args, const char *);
        XCAM_FAIL_RETURN (
            ERROR, vvalue, XCAM_RETURN_ERROR_PARAM,
            "xcam_handle(%s) set_parameters failed, param(field:%s) value should never be NULL",
            context->get_type_name (), vfield);

        params[vfield] = vvalue;
        vfield = va_arg (args, const char *);
    }
    va_end (args);

    return context->set_parameters (params);
}
#endif

SmartPtr<VideoBuffer>
external_buf_to_drm_buf (XCamVideoBuffer *buf)
{
#if 0
    // need HAVE_LIBDRM
    SmartPtr<DrmDisplay> display = DrmDisplay::instance ();
    SmartPtr<DmaVideoBuffer> dma_buf;
    SmartPtr<VideoBuffer> drm_buf;
    SmartPtr<VideoBuffer> video_buf;

    dma_buf = external_buf_to_dma_buf (buf);

    XCAM_FAIL_RETURN (
        ERROR, dma_buf.ptr (), NULL,
        "external_buf_to_drm_buf failed");

    video_buf = dma_buf;
    XCAM_ASSERT (display.ptr ());
    drm_buf = display->convert_to_drm_bo_buf (display, video_buf);
    return drm_buf;
#endif

    XCAM_LOG_ERROR ("VideoBuffer doesn't support drm buf");
    XCAM_UNUSED (buf);

    return NULL;
}

SmartPtr<VideoBuffer>
copy_external_buf_to_drm_buf (XCamHandle *handle, XCamVideoBuffer *buf)
{
    XCAM_FAIL_RETURN (ERROR, handle && buf, NULL, "xcam handle or buf can NOT be NULL");

    ContextBase *context = CONTEXT_BASE_CAST (handle);
    XCAM_FAIL_RETURN (ERROR, context, NULL, "xcam context can NOT be NULL");

    const XCamVideoBufferInfo src_info = buf->info;
    uint8_t *src = buf->map (buf);
    XCAM_FAIL_RETURN (ERROR, src, NULL, "xcam map buffer failed");

    SmartPtr<BufferPool> buf_pool = context->get_input_buffer_pool();
    XCAM_ASSERT (buf_pool.ptr ());
    SmartPtr<VideoBuffer> inbuf = buf_pool->get_buffer (buf_pool);
    XCAM_ASSERT (inbuf.ptr ());
    const VideoBufferInfo dest_info = inbuf->get_video_info ();

    VideoBufferPlanarInfo planar;
    uint8_t *dest = inbuf->map ();
    for (uint32_t idx = 0; idx < src_info.components; idx++) {
        uint8_t *p_src = src + src_info.offsets[idx];
        uint8_t *p_dest = dest + dest_info.offsets[idx];
        dest_info.get_planar_info (planar, idx);

        for (uint32_t h = 0; h < planar.height; h++) {
            memcpy (p_dest, p_src, src_info.strides[idx]);
            p_src += src_info.strides[idx];
            p_dest += dest_info.strides[idx];
        }
    }
    buf->unmap (buf);
    inbuf->unmap ();

    return inbuf;
}

XCamReturn
xcam_handle_execute (
    XCamHandle *handle, XCamVideoBuffer **buf_in, XCamVideoBuffer **buf_out)
{
    ContextBase *context = CONTEXT_BASE_CAST (handle);
    SmartPtr<VideoBuffer> input, output;

    XCAM_FAIL_RETURN (
        ERROR, context && buf_in && buf_out, XCAM_RETURN_ERROR_PARAM,
        "xcam_handle_execute failed, either of handle/buf_in/buf_out can NOT be NULL");

    XCAM_FAIL_RETURN (
        ERROR, context->is_handler_valid (), XCAM_RETURN_ERROR_PARAM,
        "context (%s) failed, handler was not initialized", context->get_type_name ());

    SmartPtr<VideoBuffer> pre, cur;
    for (int i = 0; buf_in[i] != NULL; i++) {
        if (buf_in[i]->mem_type == XCAM_MEM_TYPE_GPU) {
            cur = external_buf_to_drm_buf (buf_in[i]);
        } else {
            cur = copy_external_buf_to_drm_buf (handle, buf_in[i]);
        }
        XCAM_FAIL_RETURN (
            ERROR, cur.ptr (), XCAM_RETURN_ERROR_MEM,
            "xcam_handle(%s) execute failed, buf_in convert to DRM buffer failed.",
            context->get_type_name ());

        if (i == 0) {
            input = cur;
        } else {
            pre->attach_buffer (cur);
        }
        pre = cur;
    }

    if (*buf_out) {
        output = external_buf_to_drm_buf (*buf_out);
        XCAM_FAIL_RETURN (
            ERROR, output.ptr (), XCAM_RETURN_ERROR_MEM,
            "xcam_handle(%s) execute failed, buf_out set but convert to DRM buffer failed.",
            context->get_type_name ());
    }

    XCamReturn ret = context->execute (input, output);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS,
        ret,
        "context (%s) failed, handler execute failed", context->get_type_name ());

    if (*buf_out == NULL && output.ptr ()) {
        XCamVideoBuffer *new_buf = convert_to_external_buffer (output);
        XCAM_FAIL_RETURN (
            ERROR, new_buf, XCAM_RETURN_ERROR_MEM,
            "xcam_handle(%s) execute failed, out buffer can't convert to external buffer.",
            context->get_type_name ());
        *buf_out = new_buf;
    }

    return ret;
}
