/*
 * fisheye_dewarp.cpp - fisheye dewarp implementation
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

#include "fisheye_dewarp.h"
#include "xcam_utils.h"

namespace XCam {

FisheyeDewarp::FisheyeDewarp ()
    : _img_width (0)
    , _img_height (0)
    , _tbl_width (0)
    , _tbl_height (0)
{
}

FisheyeDewarp::~FisheyeDewarp ()
{
}

void
FisheyeDewarp::set_img_size (uint32_t width, uint32_t height)
{
    _img_width = width;
    _img_height = height;
}

void
FisheyeDewarp::set_table_size (uint32_t width, uint32_t height)
{
    _tbl_width = width;
    _tbl_height = height;
}

void
FisheyeDewarp::get_img_size (uint32_t &width, uint32_t &height)
{
    width = _img_width;
    height = _img_height;
}

void
FisheyeDewarp::get_table_size (uint32_t &width, uint32_t &height)
{
    width = _tbl_width;
    height = _tbl_height;
}

void
BowlFisheyeDewarp::set_intr_param (const IntrinsicParameter &intr_param)
{
    _intr_param = intr_param;
}

void
BowlFisheyeDewarp::set_extr_param (const ExtrinsicParameter &extr_param)
{
    _extr_param = extr_param;
}

void
BowlFisheyeDewarp::set_bowl_config (const BowlDataConfig &bowl_cfg)
{
    _bowl_cfg = bowl_cfg;
}

const IntrinsicParameter &
BowlFisheyeDewarp::get_intr_param ()
{
    return _intr_param;
}

void
BowlFisheyeDewarp::gen_table (FisheyeDewarp::MapTable &map_table)
{
    uint32_t img_w, img_h, tbl_w, tbl_h;
    get_img_size (img_w, img_h);
    get_table_size (tbl_w, tbl_h);

    XCAM_LOG_DEBUG ("fisheye-dewarp:\n table_size(%dx%d) out_size(%dx%d) "
                    "bowl(start:%.1f, end:%.1f, ground:%.2f, wall:%.2f, a:%.2f, b:%.2f, c:%.2f, center_z:%.2f)",
                    tbl_w, tbl_h, img_w, img_h,
                    _bowl_cfg.angle_start, _bowl_cfg.angle_end,
                    _bowl_cfg.ground_length, _bowl_cfg.wall_height,
                    _bowl_cfg.a, _bowl_cfg.b, _bowl_cfg.c, _bowl_cfg.center_z);

    float scale_factor_w = (float) img_w / tbl_w;
    float scale_factor_h = (float) img_h / tbl_h;

    PointFloat2 img_coord, out_pos;
    PointFloat3 world_coord, cam_coord, cam_world_coord;
    for(uint32_t row = 0; row < tbl_h; row++) {
        for(uint32_t col = 0; col < tbl_w; col++) {
            out_pos.x = col * scale_factor_w;
            out_pos.y = row * scale_factor_h;

            world_coord = bowl_view_image_to_world (_bowl_cfg, img_w, img_h, out_pos);
            cal_cam_world_coord (world_coord, cam_world_coord);
            world_coord2cam (cam_world_coord, cam_coord);
            cal_img_coord (cam_coord, img_coord);

            map_table[row * tbl_w + col] = img_coord;
        }
    }
}

void
BowlFisheyeDewarp::cal_cam_world_coord (const PointFloat3 &world_coord, PointFloat3 &cam_world_coord)
{
    Mat4f rotation_mat = generate_rotation_matrix (degree2radian (_extr_param.roll),
                         degree2radian (_extr_param.pitch), degree2radian (_extr_param.yaw));
    Mat4f rotation_tran_mat = rotation_mat;
    rotation_tran_mat (0, 3) = _extr_param.trans_x;
    rotation_tran_mat (1, 3) = _extr_param.trans_y;
    rotation_tran_mat (2, 3) = _extr_param.trans_z;

    Mat4f world_coord_mat (Vec4f(1.0f, 0.0f, 0.0f, world_coord.x),
                           Vec4f(0.0f, 1.0f, 0.0f, world_coord.y),
                           Vec4f(0.0f, 0.0f, 1.0f, world_coord.z),
                           Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    Mat4f cam_world_coord_mat = rotation_tran_mat.inverse () * world_coord_mat;

    cam_world_coord.x = cam_world_coord_mat (0, 3);
    cam_world_coord.y = cam_world_coord_mat (1, 3);
    cam_world_coord.z = cam_world_coord_mat (2, 3);
}

Mat4f
BowlFisheyeDewarp::generate_rotation_matrix (float roll, float pitch, float yaw)
{
    Mat4f matrix_x (Vec4f(1.0f, 0.0f, 0.0f, 0.0f),
                    Vec4f(0.0f, cos (roll), -sin (roll), 0.0f),
                    Vec4f(0.0f, sin (roll), cos (roll), 0.0f),
                    Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    Mat4f matrix_y (Vec4f(cos (pitch), 0.0f, sin (pitch), 0.0f),
                    Vec4f(0.0f, 1.0f, 0.0f, 0.0f),
                    Vec4f(-sin (pitch), 0.0f, cos (pitch), 0.0f),
                    Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    Mat4f matrix_z (Vec4f(cos (yaw), -sin (yaw), 0.0f, 0.0f),
                    Vec4f(sin (yaw), cos (yaw), 0.0f, 0.0f),
                    Vec4f(0.0f, 0.0f, 1.0f, 0.0f),
                    Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    return matrix_z * matrix_y * matrix_x;
}

void
BowlFisheyeDewarp::world_coord2cam (const PointFloat3 &cam_world_coord, PointFloat3 &cam_coord)
{
    cam_coord.x = -cam_world_coord.y;
    cam_coord.y = -cam_world_coord.z;
    cam_coord.z = -cam_world_coord.x;
}

void
BowlFisheyeDewarp::cal_img_coord (const PointFloat3 &cam_coord, PointFloat2 &img_coord)
{
    img_coord.x = cam_coord.x;
    img_coord.y = cam_coord.y;
}

void
PolyBowlFisheyeDewarp::cal_img_coord (const PointFloat3 &cam_coord, PointFloat2 &img_coord)
{
    float dist2center = sqrt (cam_coord.x * cam_coord.x + cam_coord.y * cam_coord.y);
    float angle = atan (cam_coord.z / dist2center);

    float p = 1;
    float poly_sum = 0;

    const IntrinsicParameter intr = get_intr_param ();

    if (dist2center != 0) {
        for (uint32_t i = 0; i < intr.poly_length; i++) {
            poly_sum += intr.poly_coeff[i] * p;
            p = p * angle;
        }

        float img_x = cam_coord.x * poly_sum / dist2center;
        float img_y = cam_coord.y * poly_sum / dist2center;

        img_coord.x = img_x * intr.c + img_y * intr.d + intr.xc;
        img_coord.y = img_x * intr.e + img_y + intr.yc;
    } else {
        img_coord.x = intr.xc;
        img_coord.y = intr.yc;
    }
} // Adopt Scaramuzza's approach to calculate image coordinates from camera coordinates

}
