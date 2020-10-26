/*
 * file.h - File class
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

#ifndef XCAM_FILE_H
#define XCAM_FILE_H

#include <xcam_std.h>

namespace XCam {

class File {
public:
    File ();
    explicit File (const char *name, const char *option);
    virtual ~File ();

    bool is_valid () const {
        return (_fp ? true : false);
    }
    bool end_of_file ();

    XCamReturn get_file_size (size_t &size);
    const char *get_file_name () const {
        return _file_name;
    }

    XCamReturn open (const char *name, const char *option);
    XCamReturn close ();
    XCamReturn rewind ();

    XCamReturn read_file (void *buf, size_t size);
    XCamReturn write_file (const void *buf, size_t size);

private:
    XCAM_DEAD_COPY (File);

protected:
    FILE     *_fp;

private:
    char     *_file_name;
    size_t    _file_size;
};

}

#endif // XCAM_FILE_H