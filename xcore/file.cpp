/*
 * file.cpp - File implementation
 *
 *  Copyright (c) 2016-2020 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "file.h"

#define INVALID_SIZE (size_t)(-1)

namespace XCam {

File::File ()
    : _fp (NULL)
    , _file_name (NULL)
    , _file_size (INVALID_SIZE)
{}

File::File (const char *name, const char *option)
    : _fp (NULL)
    , _file_name (NULL)
    , _file_size (INVALID_SIZE)
{
    open (name, option);
}

File::~File ()
{
    close ();
}

bool
File::end_of_file ()
{
    if (!is_valid ())
        return true;

    return feof (_fp);
}

XCamReturn
File::open (const char *name, const char *option)
{
    XCAM_FAIL_RETURN (
        ERROR, name != NULL && option != NULL, XCAM_RETURN_ERROR_FILE,
        "File file name or option is empty");

    close ();
    XCAM_ASSERT (!_file_name && !_fp);

    _fp = fopen (name, option);
    if (!_fp)
        return XCAM_RETURN_ERROR_FILE;

    _file_name = strndup (name, 512);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
File::close ()
{
    if (_fp) {
        fclose (_fp);
        _fp = NULL;
    }

    if (_file_name) {
        xcam_free (_file_name);
        _file_name = NULL;
    }

    _file_size = INVALID_SIZE;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
File::rewind ()
{
    if (!is_valid () || fseek (_fp, 0L, SEEK_SET) != 0)
        return XCAM_RETURN_ERROR_FILE;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
File::get_file_size (size_t &size)
{
    if (_file_size != INVALID_SIZE) {
        size = _file_size;
        return XCAM_RETURN_NO_ERROR;
    }

    fpos_t cur_pos;
    long file_size;

    if (fgetpos (_fp, &cur_pos) < 0)
        goto read_error;

    if (fseek (_fp, 0L, SEEK_END) != 0)
        goto read_error;

    if ((file_size = ftell (_fp)) <= 0)
        goto read_error;

    if (fsetpos (_fp, &cur_pos) < 0)
        goto read_error;

    _file_size = file_size;
    size = file_size;
    return XCAM_RETURN_NO_ERROR;

read_error:
    XCAM_LOG_ERROR ("File get file size failed with errno:%d", errno);
    return XCAM_RETURN_ERROR_FILE;
}

XCamReturn
File::read_file (void *buf, size_t size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (fread (buf, 1, size, _fp) != size) {
        if (end_of_file ()) {
            ret = XCAM_RETURN_BYPASS;
        } else {
            XCAM_LOG_ERROR ("File read file failed, size doesn't match");
            ret = XCAM_RETURN_ERROR_FILE;
        }
    }

    return ret;
}

XCamReturn
File::write_file (const void *buf, size_t size)
{
    if (fwrite (buf, 1, size, _fp) != size) {
        XCAM_LOG_ERROR ("File write file failed, size doesn't match");
        return XCAM_RETURN_ERROR_FILE;
    }

    return XCAM_RETURN_NO_ERROR;
}

}
