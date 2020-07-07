/*
 * context_priv.h - capi private context
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

#ifndef XCAM_CONTEXT_PRIV_H
#define XCAM_CONTEXT_PRIV_H

#include <string.h>
#include <map>
#include "xcam_utils.h"
#include "buffer_pool.h"

using namespace XCam;

enum HandleType {
    HandleTypeNone = 0,
    HandleType3DNR,
    HandleTypeWaveletNR,
    HandleTypeFisheye,
    HandleTypeDefog,
    HandleTypeDVS,
    HandleTypeStitch,
    HandleTypeStitchCL
};

typedef struct _CompareStr {
    bool operator() (const char* str1, const char* str2) const {
        return strncmp(str1, str2, 1024) < 0;
    }
} CompareStr;

typedef std::map<const char*, const char*, CompareStr> ContextParams;

class ContextBase {
public:
    virtual ~ContextBase ();

    virtual XCamReturn set_parameters (ContextParams &param_list);
    const char* get_usage () const {
        return _usage;
    }

    virtual XCamReturn init_handler () = 0;
    virtual XCamReturn uinit_handler () = 0;
    virtual bool is_handler_valid () const;

    virtual XCamReturn execute (SmartPtr<VideoBuffer> &buf_in, SmartPtr<VideoBuffer> &buf_out) = 0;

    SmartPtr<BufferPool> get_input_buffer_pool () const {
        return  _inbuf_pool;
    }
    const char* get_type_name () const;
    bool need_alloc_out_buf () const;

protected:
    ContextBase (HandleType type);

    void set_buf_pool (const SmartPtr<BufferPool> &pool);
    void set_mem_type (uint32_t type);
    void set_alloc_out_buf (bool flag);

    uint32_t get_in_width () const;
    uint32_t get_in_height () const;
    uint32_t get_out_width () const;
    uint32_t get_out_height () const;
    uint32_t get_format () const;

    void parse_value (const ContextParams &params, const char *name, uint32_t &value);

private:
    XCAM_DEAD_COPY (ContextBase);

protected:
    HandleType                       _type;
    char                            *_usage;
    SmartPtr<BufferPool>             _inbuf_pool;

    //parameters
    uint32_t                         _input_width;
    uint32_t                         _input_height;
    uint32_t                         _output_width;
    uint32_t                         _output_height;
    uint32_t                         _format;
    uint32_t                         _mem_type;
    bool                             _alloc_out_buf;
};

ContextBase *create_context (const char *name);

#endif // XCAM_CONTEXT_PRIV_H
