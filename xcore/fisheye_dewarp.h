/*
 * fisheye_dewarp.h - fisheye dewarp class
 *
 *  Copyright (c) 2017-2019 Intel Corporation
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
 * Author: Junkai Wu <junkai.wu@intel.com>
 */

#ifndef XCAM_FISHEYE_DEWARP_H
#define XCAM_FISHEYE_DEWARP_H

#include <xcam_std.h>
#include <vec_mat.h>
#include <interface/data_types.h>

namespace XCam {

class FisheyeDewarp
{
public:
    typedef std::vector<PointFloat2> MapTable;

    explicit FisheyeDewarp ();
    virtual ~FisheyeDewarp ();

    virtual void gen_table (MapTable &map_table) = 0;

    void set_in_size (uint32_t width, uint32_t height);
    void set_out_size (uint32_t width, uint32_t height);
    void set_table_size (uint32_t width, uint32_t height);

protected:
    void get_in_size (uint32_t &width, uint32_t &height);
    void get_out_size (uint32_t &width, uint32_t &height);
    void get_table_size (uint32_t &width, uint32_t &height);

private:
    XCAM_DEAD_COPY (FisheyeDewarp);

private:
    uint32_t        _in_width;
    uint32_t        _in_height;
    uint32_t        _out_width;
    uint32_t        _out_height;
    uint32_t        _tbl_width;
    uint32_t        _tbl_height;
};

class BowlFisheyeDewarp
    : public FisheyeDewarp
{
public:
    explicit BowlFisheyeDewarp () {}
    virtual ~BowlFisheyeDewarp () {}

    virtual void gen_table (FisheyeDewarp::MapTable &map_table);

    void set_intr_param (const IntrinsicParameter &intr_param);
    void set_extr_param (const ExtrinsicParameter &extr_param);
    void set_bowl_config (const BowlDataConfig &bowl_cfg);

protected:
    const IntrinsicParameter &get_intr_param ();

private:
    XCAM_DEAD_COPY (BowlFisheyeDewarp);

    virtual void cal_img_coord (const PointFloat3 &cam_coord, PointFloat2 &img_coord);

    void cal_cam_world_coord (const PointFloat3 &world_coord, PointFloat3 &cam_world_coord);
    void world_coord2cam (const PointFloat3 &cam_world_coord, PointFloat3 &cam_coord);

    Mat4f generate_rotation_matrix (float roll, float pitch, float yaw);

private:
    IntrinsicParameter        _intr_param;
    ExtrinsicParameter        _extr_param;
    BowlDataConfig            _bowl_cfg;
};

class PolyBowlFisheyeDewarp
    : public BowlFisheyeDewarp
{
public:
    explicit PolyBowlFisheyeDewarp () {}

private:
    virtual void cal_img_coord (const PointFloat3 &cam_coord, PointFloat2 &img_coord);
}; // Adopt Scaramuzza's approach to calculate image coordinates from camera coordinates

}

#endif // XCAM_FISHEYE_DEWARP_H
