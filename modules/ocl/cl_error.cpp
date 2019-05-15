/*
 * cl_error.cpp - CL errors
 *
 *  Copyright (c) 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "cl_error.h"

namespace XCam {

#define CASE(errcode) \
    case errcode: \
    { \
        snprintf (str, sizeof (str), "%d:%s", errcode, #errcode); \
        break; \
    }

const char *
error_string (cl_int code)
{
    static char str[64] = {'\0'};

    switch (code)
    {
        CASE (CL_SUCCESS);
        CASE (CL_DEVICE_NOT_FOUND);
        CASE (CL_DEVICE_NOT_AVAILABLE);
        CASE (CL_COMPILER_NOT_AVAILABLE);
        CASE (CL_MEM_OBJECT_ALLOCATION_FAILURE);
        CASE (CL_OUT_OF_RESOURCES);
        CASE (CL_OUT_OF_HOST_MEMORY);
        CASE (CL_PROFILING_INFO_NOT_AVAILABLE);
        CASE (CL_MEM_COPY_OVERLAP);
        CASE (CL_IMAGE_FORMAT_MISMATCH);
        CASE (CL_IMAGE_FORMAT_NOT_SUPPORTED);
        CASE (CL_BUILD_PROGRAM_FAILURE);
        CASE (CL_MAP_FAILURE);
        CASE (CL_MISALIGNED_SUB_BUFFER_OFFSET);
        CASE (CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        CASE (CL_COMPILE_PROGRAM_FAILURE);
        CASE (CL_LINKER_NOT_AVAILABLE);
        CASE (CL_LINK_PROGRAM_FAILURE);
        CASE (CL_DEVICE_PARTITION_FAILED);
        CASE (CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        CASE (CL_INVALID_VALUE);
        CASE (CL_INVALID_DEVICE_TYPE);
        CASE (CL_INVALID_PLATFORM);
        CASE (CL_INVALID_DEVICE);
        CASE (CL_INVALID_CONTEXT);
        CASE (CL_INVALID_QUEUE_PROPERTIES);
        CASE (CL_INVALID_COMMAND_QUEUE);
        CASE (CL_INVALID_HOST_PTR);
        CASE (CL_INVALID_MEM_OBJECT);
        CASE (CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        CASE (CL_INVALID_IMAGE_SIZE);
        CASE (CL_INVALID_SAMPLER);
        CASE (CL_INVALID_BINARY);
        CASE (CL_INVALID_BUILD_OPTIONS);
        CASE (CL_INVALID_PROGRAM);
        CASE (CL_INVALID_PROGRAM_EXECUTABLE);
        CASE (CL_INVALID_KERNEL_NAME);
        CASE (CL_INVALID_KERNEL_DEFINITION);
        CASE (CL_INVALID_KERNEL);
        CASE (CL_INVALID_ARG_INDEX);
        CASE (CL_INVALID_ARG_VALUE);
        CASE (CL_INVALID_ARG_SIZE);
        CASE (CL_INVALID_KERNEL_ARGS);
        CASE (CL_INVALID_WORK_DIMENSION);
        CASE (CL_INVALID_WORK_GROUP_SIZE);
        CASE (CL_INVALID_WORK_ITEM_SIZE);
        CASE (CL_INVALID_GLOBAL_OFFSET);
        CASE (CL_INVALID_EVENT_WAIT_LIST);
        CASE (CL_INVALID_EVENT);
        CASE (CL_INVALID_OPERATION);
        CASE (CL_INVALID_GL_OBJECT);
        CASE (CL_INVALID_BUFFER_SIZE);
        CASE (CL_INVALID_MIP_LEVEL);
        CASE (CL_INVALID_GLOBAL_WORK_SIZE);
        CASE (CL_INVALID_PROPERTY);
        CASE (CL_INVALID_IMAGE_DESCRIPTOR);
        CASE (CL_INVALID_COMPILER_OPTIONS);
        CASE (CL_INVALID_LINKER_OPTIONS);
        CASE (CL_INVALID_DEVICE_PARTITION_COUNT);
        CASE (CL_INVALID_PIPE_SIZE);
        CASE (CL_INVALID_DEVICE_QUEUE);
#if CL_VERSION_2_2
        CASE (CL_INVALID_SPEC_ID);
        CASE (CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif
        default:
            snprintf (str, sizeof (str), "unknown code:%d", code);
            XCAM_LOG_ERROR ("%s", str);
    }

    return str;
}

}
