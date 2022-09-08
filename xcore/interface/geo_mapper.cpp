/*
 * geo_mapper.cpp - geometry mapper implementation
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

#include "geo_mapper.h"

namespace XCam {

GeoMapper::GeoMapper ()
    : _out_width (0)
    , _out_height (0)
    , _std_out_width (0)
    , _std_out_height (0)
    , _lut_width (0)
    , _lut_height (0)
    , _factor_x (0.0f)
    , _factor_y (0.0f)
    , _thread_x (12)
    , _thread_y (8)
{}

GeoMapper::~GeoMapper ()
{
}

bool
GeoMapper::set_factors (float x, float y)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f), false,
        "GeoMapper set factors failed, x:%.3f, h:%.3f", x, y);
    _factor_x = x;
    _factor_y = y;

    return true;
}

void
GeoMapper::get_factors (float &x, float &y) const
{
    x = _factor_x;
    y = _factor_y;
}

bool
GeoMapper::set_output_size (uint32_t width, uint32_t height)
{
    XCAM_FAIL_RETURN (
        ERROR, width && height, false,
        "GeoMapper set output size failed, w:%d, h:%d", width, height);

    _out_width = width;
    _out_height = height;
    return true;
}

void
GeoMapper::get_output_size (uint32_t &width, uint32_t &height) const
{
    width = _out_width;
    height = _out_height;
}

bool
GeoMapper::set_std_output_size (uint32_t width, uint32_t height)
{
    XCAM_FAIL_RETURN (
        ERROR, width && height, false,
        "GeoMapper set standard output size failed, w:%d, h:%d", width, height);

    _std_out_width = width;
    _std_out_height = height;

    return true;
}

void
GeoMapper::get_std_output_size (uint32_t &width, uint32_t &height) const
{
    width = _std_out_width;
    height = _std_out_height;
}

bool
GeoMapper::set_thread_count (uint32_t x, uint32_t y)
{
    XCAM_FAIL_RETURN (
        ERROR, x && y, false,
        "GeoMapper set thread count failed, x:%d, y:%d", x, y);

    _thread_x = x;
    _thread_y = y;

    return true;
}

void
GeoMapper::get_thread_count (uint32_t &x, uint32_t &y) const
{
    x = _thread_x;
    y = _thread_y;
}

bool
GeoMapper::auto_calculate_factors (uint32_t lut_w, uint32_t lut_h)
{
    XCAM_FAIL_RETURN (
        ERROR, _std_out_width > 1 && _std_out_height > 1, false,
        "GeoMapper auto calculate factors failed, standard output size was not set, w:%d, h:%d",
        _std_out_width, _std_out_height);
    XCAM_FAIL_RETURN (
        ERROR, lut_w > 1 && lut_h > 1, false,
        "GeoMapper auto calculate factors failed, lookuptable size need > 1. but set with w:%d, h:%d",
        lut_w, lut_h);

    _factor_x = (_std_out_width - 1.0f) / (lut_w - 1.0f);
    _factor_y = (_std_out_height - 1.0f) / (lut_h - 1.0f);

    return true;
}

}
