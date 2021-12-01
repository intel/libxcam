/*
 * data_types.h - data types in interface
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_INTERFACE_DATA_TYPES_H
#define XCAM_INTERFACE_DATA_TYPES_H

#include <xcam_std.h>
#include <vec_mat.h>

namespace XCam {

enum CamModel {
    CamA2C1080P = 0,
    CamB4C1080P,
    CamC3C8K,
    CamC6C8K,
    CamD3C8K
};

enum FisheyeDewarpMode {
    DewarpSphere = 0,
    DewarpBowl
};

enum FeatureMatchMode {
    FMNone = 0,
    FMDefault,
    FMCluster,
    FMCapi
};

enum FeatureMatchStatus {
    FMStatusWholeWay = 0,
    FMStatusHalfWay,
    FMStatusFMFirst
};

enum GeoMapScaleMode {
    ScaleSingleConst = 0,
    ScaleDualConst,
    ScaleDualCurve
};

struct Rect {
    int32_t pos_x, pos_y;
    int32_t width, height;

    Rect () : pos_x (0), pos_y (0), width (0), height (0) {}
    Rect (int32_t x, int32_t y, int32_t w, int32_t h) : pos_x (x), pos_y (y), width (w), height (h) {}
};

struct ImageCropInfo {
    uint32_t left;
    uint32_t right;
    uint32_t top;
    uint32_t bottom;

    ImageCropInfo () : left (0), right (0), top (0), bottom (0) {}
};

#define XCAM_INTRINSIC_MAX_POLY_SIZE 16

// current intrinsic parameters definition from Scaramuzza's approach
struct IntrinsicParameter {
    uint32_t width;
    uint32_t height;
    float cx;
    float cy;
    float fx;
    float fy;

    float fov;
    float skew;

    float c;
    float d;
    float e;
    uint32_t poly_length;

    float poly_coeff[XCAM_INTRINSIC_MAX_POLY_SIZE];

    bool  flip;

    IntrinsicParameter ()
        : width (0), height (0), cx (0.0f), cy (0.0f), fx (0.0), fy (0.0),
          fov (0.0), skew (0.0),
          c (0.0f), d (0.0f), e (0.0f), poly_length (0), flip (false)
    {
        xcam_mem_clear (poly_coeff);
    }
};

struct ExtrinsicParameter {
    float trans_x;
    float trans_y;
    float trans_z;

    // angle degree
    float roll;
    float pitch;
    float yaw;

    ExtrinsicParameter ()
        : trans_x (0.0f), trans_y (0.0f), trans_z (0.0f)
        , roll (0.0f), pitch (0.0f), yaw (0.0f)
    {}
};

struct CalibrationInfo {
    uint32_t camera_id;
    ExtrinsicParameter extrinsic;
    IntrinsicParameter intrinsic;

    CalibrationInfo ()
        : camera_id (0)
    {}
};

struct CameraInfo {
    float             round_angle_start;
    float             angle_range;
    CalibrationInfo   calibration;

    CameraInfo ()
        : round_angle_start (0.0f)
        , angle_range (0.0f)
    {}
};

struct FisheyeInfo : CalibrationInfo {
    float radius;
    float distort_coeff[4];
    float c_coeff[4];
    uint32_t cam_model;

    FisheyeInfo ()
        : radius (0.0f)
    {
        xcam_mem_clear (distort_coeff);
        xcam_mem_clear (c_coeff);
    }
    bool is_valid () const {
        return intrinsic.fov >= 1.0f && radius >= 1.0f;
    }
};

template <typename T>
struct Point2DT {
    T x, y;
    Point2DT () : x (0), y(0) {}
    Point2DT (const T px, const T py) : x (px), y(py) {}
};

template <typename T>
struct Point3DT {
    T x, y, z;
    Point3DT () : x (0), y(0), z(0) {}
    Point3DT (const T px, const T py, const T pz) : x (px), y(py), z(pz) {}
};

typedef Point2DT<int32_t> PointInt2;
typedef Point2DT<float> PointFloat2;

typedef Point3DT<int32_t> PointInt3;
typedef Point3DT<float> PointFloat3;

/*
 * Ellipsoid model
 *  x^2 / a^2 + y^2 / b^2 + (z-center_z)^2 / c^2 = 1
 * ground : z = 0
 * x_axis : front direction
 * y_axis : left direction
 * z_axis : up direction
 * wall_height : bowl height inside of view
 * ground_length: left direction distance from ellipsoid bottom edge to nearest side of the car in the view
 */
struct BowlDataConfig {
    float a, b, c;
    float angle_start, angle_end; // angle degree

    // unit mm
    float center_z;
    float wall_height;
    float ground_length;

    BowlDataConfig ()
        : a (6060.0f), b (4388.0f), c (3003.4f)
        , angle_start (90.0f), angle_end (270.0f)
        , center_z (1500.0f)
        , wall_height (3000.0f)
        , ground_length (2801.0f)
    {
        XCAM_ASSERT (fabs(center_z) <= c);
        XCAM_ASSERT (a > 0.0f && b > 0.0f && c > 0.0f);
        XCAM_ASSERT (wall_height >= 0.0f && ground_length >= 0.0f);
        XCAM_ASSERT (ground_length <= b * sqrt(1.0f - center_z * center_z / (c * c)));
        XCAM_ASSERT (wall_height <= center_z + c);
    }
};

}

#endif //XCAM_INTERFACE_DATA_TYPES_H
