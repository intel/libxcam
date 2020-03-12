/*
 * soft_stitcher.cpp - soft stitcher implementation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "fisheye_dewarp.h"
#include "soft_stitcher.h"
#include "soft_blender.h"
#include "soft_geo_mapper.h"
#include "soft_video_buf_allocator.h"
#include "interface/feature_match.h"
#include "soft_copy_task.h"
#include "xcam_utils.h"
#include <map>

#define ENABLE_FEATURE_MATCH HAVE_OPENCV

#if ENABLE_FEATURE_MATCH
#include "ocv/cv_capi_feature_match.h"
#ifndef ANDROID
#include <opencv2/core/ocl.hpp>
#endif
#endif

#define SOFT_STITCHER_ALIGNMENT_X 8
#define SOFT_STITCHER_ALIGNMENT_Y 4

#define MAP_FACTOR_X  16
#define MAP_FACTOR_Y  16

#define DUMP_STITCHER 0

namespace XCam {

#if DUMP_STITCHER
static void
stitcher_dump_buf (const SmartPtr<VideoBuffer> buf, uint32_t idx, const char *prefix)
{
    XCAM_ASSERT (prefix);
    char name[256];
    snprintf (name, 256, "%s-%d", prefix, idx);
    dump_buf_perfix_path (buf, name);
}
#else
static void stitcher_dump_buf (const SmartPtr<VideoBuffer> buf, ...) {
    XCAM_UNUSED (buf);
}
#endif

static inline bool complete_stitch (Stitcher *stitcher, uint32_t frame_count)
{
    return (stitcher->get_fm_mode () == FMNone ||
            stitcher->get_fm_status () != FMStatusFMFirst ||
            frame_count >= stitcher->get_fm_frames ());
}

namespace SoftStitcherPriv {

DECLARE_HANDLER_CALLBACK (CbGeoMap, SoftStitcher, geomap_done);
DECLARE_HANDLER_CALLBACK (CbBlender, SoftStitcher, blender_done);
DECLARE_WORK_CALLBACK (CbCopyTask, SoftStitcher, copy_task_done);

struct BlenderParam
    : SoftBlender::BlenderParam
{
    SmartPtr<SoftStitcher::StitcherParam>  stitch_param;
    uint32_t idx;

    BlenderParam (
        uint32_t i,
        const SmartPtr<VideoBuffer> &in0,
        const SmartPtr<VideoBuffer> &in1,
        const SmartPtr<VideoBuffer> &out)
        : SoftBlender::BlenderParam (in0, in1, out)
        , idx (i)
    {}
};

typedef std::map<void*, SmartPtr<BlenderParam>> BlenderParams;
typedef std::map<void*, int32_t> BlendCopyTaskNums;

struct HandlerParam
    : ImageHandler::Parameters
{
    SmartPtr<SoftStitcher::StitcherParam>  stitch_param;
    uint32_t idx;

    HandlerParam (uint32_t i)
        : idx (i)
    {}
};

struct StitcherCopyArgs
    : XCamSoftTasks::CopyTask::Args
{
    uint32_t idx;

    StitcherCopyArgs (
        uint32_t i,
        const SmartPtr<ImageHandler::Parameters> &param)
        : XCamSoftTasks::CopyTask::Args (param)
        , idx (i)
    {}
};

struct Factor {
    float x, y;

    Factor () : x (1.0f), y (1.0f) {}
    void reset () {
        x = 1.0f;
        y = 1.0f;
    }
};

struct Overlap {
    SmartPtr<FeatureMatch>       matcher;
    SmartPtr<SoftBlender>        blender;
    BlenderParams                param_map;

    SmartPtr<BlenderParam> find_blender_param_in_map (
        const SmartPtr<SoftStitcher::StitcherParam> &key,
        const uint32_t idx);
};

struct FisheyeMap {
    SmartPtr<SoftGeoMapper>      mapper;
    SmartPtr<BufferPool>         buf_pool;
    FisheyeDewarpMode            dewarp_mode;
    FisheyeInfo                  fisheye_info;
    Factor                       left_match_factor, right_match_factor;

    XCamReturn set_map_table (
        SoftStitcher *stitcher, const Stitcher::RoundViewSlice &view_slice, uint32_t cam_idx);
};

struct Copier {
    SmartPtr<XCamSoftTasks::CopyTask>    copy_task;
    Stitcher::CopyArea                   copy_area;

    XCamReturn start_copy_task (
        const SmartPtr<ImageHandler::Parameters> &param,
        const uint32_t idx, const SmartPtr<VideoBuffer> &buf);
};
typedef std::vector<Copier>    Copiers;

class StitcherImpl {
    friend class XCam::SoftStitcher;

public:
    StitcherImpl (SoftStitcher *handler)
        : _stitcher (handler)
        , _pixel_format (V4L2_PIX_FMT_NV12)
    {}

    XCamReturn init_config (uint32_t count);

    bool remove_task_count (const SmartPtr<SoftStitcher::StitcherParam> &param);
    int32_t dec_task_count (const SmartPtr<SoftStitcher::StitcherParam> &param);

    XCamReturn start_geomap_works (const SmartPtr<SoftStitcher::StitcherParam> &param);
    XCamReturn start_task_count (const SmartPtr<SoftStitcher::StitcherParam> &param);
    XCamReturn start_overlap_tasks (
        const SmartPtr<SoftStitcher::StitcherParam> &param,
        const uint32_t idx, const SmartPtr<VideoBuffer> &buf);
    XCamReturn start_copy_tasks (
        const SmartPtr<SoftStitcher::StitcherParam> &param,
        const uint32_t idx, const SmartPtr<VideoBuffer> &buf);

    XCamReturn start_overlap_task (uint32_t idx, const SmartPtr<BlenderParam> &param);
    XCamReturn stop ();

    XCamReturn gen_geomap_table ();
    XCamReturn start_feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf, const uint32_t idx);

    bool get_and_reset_feature_match_factors (uint32_t idx, Factor &left, Factor &right);
    void set_pixel_format (uint32_t format) {
        _pixel_format = format;
    };
    uint32_t get_pixel_format () const {
        return _pixel_format;
    };

private:
    SmartPtr<SoftGeoMapper> create_geo_mapper (const Stitcher::RoundViewSlice &view_slice);

    XCamReturn init_fisheye (uint32_t idx);
    XCamReturn init_blender (uint32_t idx);
    XCamReturn init_copier (Stitcher::CopyArea area);
    bool init_geomap_factors (uint32_t idx);
    XCamReturn create_copier (Stitcher::CopyArea area);

    void calc_factors (
        const uint32_t &idx, const Factor &last_left_factor, const Factor &last_right_factor,
        Factor &cur_left, Factor &cur_right);

    void init_feature_match (uint32_t idx);

private:
    StitchInfo              _stitch_info;
    FisheyeMap              _fisheye [XCAM_STITCH_MAX_CAMERAS];
    Overlap                 _overlaps [XCAM_STITCH_MAX_CAMERAS];
    Copiers                 _copiers;
    SmartPtr<BufferPool>    _geomap_pool;

    Mutex                   _map_mutex;
    BlendCopyTaskNums       _task_counts;

    SoftStitcher           *_stitcher;
    uint32_t               _pixel_format;
};

XCamReturn
FisheyeMap::set_map_table (
    SoftStitcher *stitcher, const Stitcher::RoundViewSlice &view_slice, uint32_t cam_idx)
{
    SmartPtr<FisheyeDewarp> dewarper;
    if(dewarp_mode == DewarpBowl) {
        BowlDataConfig bowl = stitcher->get_bowl_config ();
        bowl.angle_start = view_slice.hori_angle_start;
        bowl.angle_end = format_angle (view_slice.hori_angle_start + view_slice.hori_angle_range);
        if (bowl.angle_end < bowl.angle_start)
            bowl.angle_start -= 360.0f;

        XCAM_LOG_DEBUG (
            "soft-stitcher:%s camera(idx:%d) info(angle start:%.2f, range:%.2f), bowl_info(angle start%.2f, end:%.2f)",
            XCAM_STR (stitcher->get_name ()), cam_idx,
            view_slice.hori_angle_start, view_slice.hori_angle_range, bowl.angle_start, bowl.angle_end);

        CameraInfo cam_info;
        stitcher->get_camera_info (cam_idx, cam_info);

        SmartPtr<PolyBowlFisheyeDewarp> fd = new PolyBowlFisheyeDewarp ();
        fd->set_intr_param (cam_info.calibration.intrinsic);
        fd->set_extr_param (cam_info.calibration.extrinsic);
        fd->set_bowl_config (bowl);
        dewarper = fd;
    } else {
        float max_dst_latitude = (fisheye_info.wide_angle > 180.0f) ? 180.0f : fisheye_info.wide_angle;
        float max_dst_longitude = max_dst_latitude * view_slice.width / view_slice.height;

        SmartPtr<SphereFisheyeDewarp> fd = new SphereFisheyeDewarp ();
        fd->set_fisheye_info (fisheye_info);
        fd->set_dst_range (max_dst_longitude, max_dst_latitude);
        dewarper = fd;
    }
    XCAM_FAIL_RETURN (
        ERROR, dewarper.ptr (), XCAM_RETURN_ERROR_MEM,
        "soft-stitcher:%s fisheye dewarper is NULL", XCAM_STR (stitcher->get_name ()));

    dewarper->set_out_size (view_slice.width, view_slice.height);

    uint32_t table_width = view_slice.width / MAP_FACTOR_X;
    table_width = XCAM_ALIGN_UP (table_width, 4);
    uint32_t table_height = view_slice.height / MAP_FACTOR_Y;
    table_height = XCAM_ALIGN_UP (table_height, 2);
    dewarper->set_table_size (table_width, table_height);

    FisheyeDewarp::MapTable map_table (table_width * table_height);
    dewarper->gen_table (map_table);

    XCAM_FAIL_RETURN (
        ERROR, mapper->set_lookup_table (map_table.data (), table_width, table_height), XCAM_RETURN_ERROR_UNKNOWN,
        "soft-stitcher:%s set fisheye geomap lookup table failed", XCAM_STR (stitcher->get_name ()));

    return XCAM_RETURN_NO_ERROR;
}

void
StitcherImpl::calc_factors (
    const uint32_t &idx, const Factor &last_left_factor, const Factor &last_right_factor,
    Factor &cur_left, Factor &cur_right)
{
    Factor match_left_factor, match_right_factor;
    get_and_reset_feature_match_factors (idx, match_left_factor, match_right_factor);

    cur_left.x = last_left_factor.x * match_left_factor.x;
    cur_left.y = last_left_factor.y * match_left_factor.y;
    cur_right.x = last_right_factor.x * match_right_factor.x;
    cur_right.y = last_right_factor.y * match_right_factor.y;
}

bool
StitcherImpl::init_geomap_factors (uint32_t idx)
{
    XCAM_FAIL_RETURN (
        ERROR, _fisheye[idx].mapper.ptr (), false,
        "FisheyeDewarp geomap handler is empty");

    Factor last_left_factor, last_right_factor, cur_left, cur_right;
    if (_stitcher->get_scale_mode () == ScaleSingleConst) {
        Factor unify_factor;
        _fisheye[idx].mapper->get_factors (unify_factor.x, unify_factor.y);
        if (XCAM_DOUBLE_EQUAL_AROUND (unify_factor.x, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (unify_factor.y, 0.0f)) { // not started.
            return true;
        }
        last_left_factor = last_right_factor = unify_factor;

        calc_factors (idx, last_left_factor, last_right_factor, cur_left, cur_right);
        unify_factor.x = (cur_left.x + cur_right.x) / 2.0f;
        unify_factor.y = (cur_left.y + cur_right.y) / 2.0f;

        _fisheye[idx].mapper->set_factors (unify_factor.x, unify_factor.y);
    } else {
        SmartPtr<SoftDualConstGeoMapper> mapper = _fisheye[idx].mapper.dynamic_cast_ptr<SoftDualConstGeoMapper> ();
        XCAM_ASSERT (mapper.ptr ());
        mapper->get_left_factors (last_left_factor.x, last_left_factor.y);
        mapper->get_right_factors (last_right_factor.x, last_right_factor.y);
        if (XCAM_DOUBLE_EQUAL_AROUND (last_left_factor.x, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (last_left_factor.y, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (last_right_factor.y, 0.0f) ||
                XCAM_DOUBLE_EQUAL_AROUND (last_right_factor.y, 0.0f)) { // not started.
            return true;
        }

        calc_factors (idx, last_left_factor, last_right_factor, cur_left, cur_right);

        mapper->set_left_factors (cur_left.x, cur_left.y);
        mapper->set_right_factors (cur_right.x, cur_right.y);
    }

    return true;
}

bool
StitcherImpl::get_and_reset_feature_match_factors (uint32_t idx, Factor &left, Factor &right)
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    XCAM_FAIL_RETURN (
        ERROR, idx < cam_num, false,
        "get geomap factor failed, idx(%d) > camera_num(%d)", idx, cam_num);

    SmartLock locker (_map_mutex);
    left = _fisheye[idx].left_match_factor;
    right = _fisheye[idx].right_match_factor;

    _fisheye[idx].left_match_factor.reset ();
    _fisheye[idx].right_match_factor.reset ();
    return true;
}

SmartPtr<SoftGeoMapper>
StitcherImpl::create_geo_mapper (const Stitcher::RoundViewSlice &view_slice)
{
    SmartPtr<SoftGeoMapper> mapper;
    if (_stitcher->get_scale_mode () == ScaleSingleConst)
        mapper = new SoftGeoMapper ("stitcher_remapper");
    else if (_stitcher->get_scale_mode () == ScaleDualConst)
        mapper = new SoftDualConstGeoMapper ("stitcher_dualconst_remapper");
    else {
        SmartPtr<SoftDualCurveGeoMapper> geomap = new SoftDualCurveGeoMapper ("stitcher_dualcurve_remapper");
        XCAM_ASSERT (geomap.ptr ());

        BowlDataConfig bowl = _stitcher->get_bowl_config ();
        float scaled_height = (bowl.wall_height + bowl.ground_length / 2.0f) /
                              (bowl.wall_height + bowl.ground_length) * view_slice.height;

        geomap->set_scaled_height (scaled_height);
        mapper = geomap;
    }

    XCAM_ASSERT (mapper.ptr ());
    return mapper;
}

XCamReturn
StitcherImpl::init_fisheye (uint32_t idx)
{
    FisheyeMap &fisheye = _fisheye[idx];
    fisheye.dewarp_mode = _stitcher->get_dewarp_mode ();
    if (fisheye.dewarp_mode == DewarpSphere) {
        fisheye.fisheye_info = _stitch_info.fisheye_info[idx];
    }

    Stitcher::RoundViewSlice view_slice = _stitcher->get_round_view_slice (idx);

    SmartPtr<ImageHandler::Callback> geomap_cb = new CbGeoMap (_stitcher);
    fisheye.mapper = create_geo_mapper (view_slice);
    fisheye.mapper->set_callback (geomap_cb);

    VideoBufferInfo buf_info;
    uint32_t pixel_format = get_pixel_format ();
    buf_info.init (
        pixel_format, view_slice.width, view_slice.height,
        XCAM_ALIGN_UP (view_slice.width, SOFT_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (view_slice.height, SOFT_STITCHER_ALIGNMENT_Y));

    SmartPtr<BufferPool> pool = new SoftVideoBufAllocator (buf_info);
    XCAM_ASSERT (pool.ptr ());
    fisheye.buf_pool = pool;
    XCAM_FAIL_RETURN (
        ERROR, fisheye.buf_pool->reserve (2), XCAM_RETURN_ERROR_MEM,
        "stitcher:%s reserve geomap buffer pool(w:%d,h:%d) failed",
        XCAM_STR (_stitcher->get_name ()), buf_info.width, buf_info.height);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::create_copier (Stitcher::CopyArea area)
{
    XCAM_FAIL_RETURN (
        ERROR,
        area.in_idx != INVALID_INDEX &&
        area.in_area.width == area.out_area.width && area.in_area.height == area.out_area.height,
        XCAM_RETURN_ERROR_PARAM,
        "stitcher: copy area (idx:%d) is invalid", area.in_idx);

    SmartPtr<Worker::Callback> copy_cb = new CbCopyTask (_stitcher);
    XCAM_ASSERT (copy_cb.ptr ());

    Copier copier;
    copier.copy_task = new XCamSoftTasks::CopyTask (copy_cb);
    XCAM_ASSERT (copier.copy_task.ptr ());
    copier.copy_area = area;
    _copiers.push_back (copier);

    return XCAM_RETURN_NO_ERROR;
}

void
StitcherImpl::init_feature_match (uint32_t idx)
{
#if ENABLE_FEATURE_MATCH

#ifndef ANDROID
    FeatureMatchMode fm_mode = _stitcher->get_fm_mode ();
    if (fm_mode == FMNone)
        return ;
    else if (fm_mode == FMDefault)
        _overlaps[idx].matcher = FeatureMatch::create_default_feature_match ();
    else if (fm_mode == FMCluster)
        _overlaps[idx].matcher = FeatureMatch::create_cluster_feature_match ();
    else if (fm_mode == FMCapi)
        _overlaps[idx].matcher = FeatureMatch::create_capi_feature_match ();
    else {
        XCAM_LOG_ERROR ("unsupported FeatureMatchMode: %d", fm_mode);
        XCAM_ASSERT (false);
    }
#else
    _overlaps[idx].matcher = new CVCapiFeatureMatch;
#endif
    XCAM_ASSERT (_overlaps[idx].matcher.ptr ());

    _overlaps[idx].matcher->set_config (_stitcher->get_fm_config ());
    _overlaps[idx].matcher->set_fm_index (idx);

    const BowlDataConfig bowl = _stitcher->get_bowl_config ();
    const Stitcher::ImageOverlapInfo &info = _stitcher->get_overlap (idx);
    Rect left_ovlap = info.left;
    Rect right_ovlap = info.right;

    if (_stitcher->get_dewarp_mode () == DewarpSphere) {
        const FMRegionRatio &ratio = _stitcher->get_fm_region_ratio ();

        left_ovlap.pos_y = left_ovlap.height * ratio.pos_y;
        left_ovlap.height = left_ovlap.height * ratio.height;
        right_ovlap.pos_y = left_ovlap.pos_y;
        right_ovlap.height = left_ovlap.height;
    } else {
        left_ovlap.pos_y = 0;
        left_ovlap.height = int32_t (bowl.wall_height / (bowl.wall_height + bowl.ground_length) * left_ovlap.height);
        right_ovlap.pos_y = 0;
        right_ovlap.height = left_ovlap.height;
    }
    _overlaps[idx].matcher->set_crop_rect (left_ovlap, right_ovlap);
#else
    XCAM_LOG_ERROR ("FeatureMatch unsupported");
    XCAM_ASSERT (false);
#endif
}

XCamReturn
StitcherImpl::init_blender (uint32_t idx)
{
    _overlaps[idx].blender = create_soft_blender ().dynamic_cast_ptr<SoftBlender>();
    XCAM_ASSERT (_overlaps[idx].blender.ptr ());

    _overlaps[idx].blender->set_pyr_levels (_stitcher->get_blend_pyr_levels ());

    uint32_t out_width, out_height;
    _stitcher->get_output_size (out_width, out_height);
    _overlaps[idx].blender->set_output_size (out_width, out_height);

    const Stitcher::ImageOverlapInfo overlap_info = _stitcher->get_overlap (idx);
    Stitcher::ImageOverlapInfo overlap = overlap_info;
    if (_stitcher->get_dewarp_mode () == DewarpSphere && _stitch_info.merge_width[idx] > 0) {
        uint32_t specific_merge_width = _stitch_info.merge_width[idx];
        XCAM_ASSERT (uint32_t (overlap.left.width) >= specific_merge_width);

        uint32_t ext_width = (overlap.left.width - specific_merge_width) / 2;
        ext_width = XCAM_ALIGN_UP (ext_width, SOFT_STITCHER_ALIGNMENT_X);

        overlap.left.pos_x += ext_width;
        overlap.left.width = specific_merge_width;
        overlap.right.pos_x += ext_width;
        overlap.right.width = specific_merge_width;
        overlap.out_area.pos_x += ext_width;
        overlap.out_area.width = specific_merge_width;
    }
    _overlaps[idx].blender->set_merge_window (overlap.out_area);
    _overlaps[idx].blender->set_input_valid_area (overlap.left, 0);
    _overlaps[idx].blender->set_input_valid_area (overlap.right, 1);
    _overlaps[idx].blender->set_input_merge_area (overlap.left, 0);
    _overlaps[idx].blender->set_input_merge_area (overlap.right, 1);

    SmartPtr<ImageHandler::Callback> blender_cb = new CbBlender (_stitcher);
    XCAM_ASSERT (blender_cb.ptr ());
    _overlaps[idx].blender->set_callback (blender_cb);

    _overlaps[idx].param_map.clear ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_copier (Stitcher::CopyArea area)
{
    if (_stitcher->get_dewarp_mode () == DewarpSphere && _stitch_info.merge_width[area.in_idx] > 0) {
        uint32_t specific_merge_width = _stitch_info.merge_width[area.in_idx];
        const Stitcher::ImageOverlapInfo overlap_info = _stitcher->get_overlap (area.in_idx);
        XCAM_ASSERT (uint32_t (overlap_info.left.width) >= specific_merge_width);

        uint32_t ext_width = (overlap_info.left.width - specific_merge_width) / 2;
        ext_width = XCAM_ALIGN_UP (ext_width, SOFT_STITCHER_ALIGNMENT_X);

        area.in_area.width = (area.in_idx == 0) ?
                             (area.in_area.width + ext_width) : (area.in_area.width + ext_width * 2);
        area.out_area.width = area.in_area.width;
        if (area.out_area.pos_x > 0) {
            area.in_area.pos_x -= ext_width;
            area.out_area.pos_x -= ext_width;
        }
    }

    XCAM_LOG_DEBUG ("soft-stitcher:copy area (idx:%d) input area(%d, %d, %d, %d) output area(%d, %d, %d, %d)",
                    area.in_idx,
                    area.in_area.pos_x, area.in_area.pos_y, area.in_area.width, area.in_area.height,
                    area.out_area.pos_x, area.out_area.pos_y, area.out_area.width, area.out_area.height);

    XCamReturn ret = create_copier (area);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher::%s create copier failed, idx:%d.", XCAM_STR (_stitcher->get_name ()), area.in_idx);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_config (uint32_t count)
{
    if (_stitcher->get_dewarp_mode () == DewarpSphere) {
        _stitch_info = _stitcher->get_stitch_info ();
    }

    for (uint32_t i = 0; i < count; ++i) {
        XCamReturn ret = init_fisheye (i);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft-stitcher:%s init fisheye failed, idx:%d.", XCAM_STR (_stitcher->get_name ()), i);

#if ENABLE_FEATURE_MATCH
        init_feature_match (i);
#endif

        init_blender (i);
    }

    Stitcher::CopyAreaArray areas = _stitcher->get_copy_area ();
    uint32_t size = areas.size ();
    for (uint32_t i = 0; i < size; ++i) {
        XCAM_ASSERT (areas[i].in_idx < size);

        XCamReturn ret = init_copier (areas[i]);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft-stitcher:%s init copyer failed, idx:%d.", XCAM_STR (_stitcher->get_name ()), areas[i].in_idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

bool
StitcherImpl::remove_task_count (const SmartPtr<SoftStitcher::StitcherParam> &param)
{
    XCAM_ASSERT (param.ptr ());
    SmartLock locker (_map_mutex);
    BlendCopyTaskNums::iterator i = _task_counts.find (param.ptr ());
    if (i == _task_counts.end ())
        return false;

    _task_counts.erase (i);

    return true;
}

int32_t
StitcherImpl::dec_task_count (const SmartPtr<SoftStitcher::StitcherParam> &param)
{
    XCAM_ASSERT (param.ptr ());
    SmartLock locker (_map_mutex);
    BlendCopyTaskNums::iterator i = _task_counts.find (param.ptr ());
    if (i == _task_counts.end ())
        return -1;

    int32_t &count = i->second;
    --count;
    if (count > 0)
        return count;

    XCAM_ASSERT (count == 0);
    _task_counts.erase (i);

    return 0;
}

XCamReturn
StitcherImpl::gen_geomap_table ()
{
    uint32_t camera_num = _stitcher->get_camera_num ();
    for (uint32_t i = 0; i < camera_num; ++i) {
        const Stitcher::RoundViewSlice view_slice = _stitcher->get_round_view_slice (i);
        _fisheye[i].mapper->set_output_size (view_slice.width, view_slice.height);

        XCamReturn ret = _fisheye[i].set_map_table (_stitcher, view_slice, i);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "stitcher:%s generate geomap table failed, idx:%d", XCAM_STR (_stitcher->get_name ()), i);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_geomap_works (const SmartPtr<SoftStitcher::StitcherParam> &param)
{
    uint32_t camera_num = _stitcher->get_camera_num ();
    Factor cur_left, cur_right;

    for (uint32_t i = 0; i < camera_num; ++i) {
        SmartPtr<VideoBuffer> out_buf = _fisheye[i].buf_pool->get_buffer ();
        SmartPtr<HandlerParam> geomap_params = new HandlerParam (i);
        geomap_params->in_buf = param->in_bufs[i];
        geomap_params->out_buf = out_buf;
        geomap_params->stitch_param = param;

        init_geomap_factors (i);
        XCamReturn ret = _fisheye[i].mapper->execute_buffer (geomap_params, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft-stitcher:%s fisheye geomap buffer failed", XCAM_STR (_stitcher->get_name ()));
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<BlenderParam>
Overlap::find_blender_param_in_map (
    const SmartPtr<SoftStitcher::StitcherParam> &key,
    const uint32_t idx)
{
    SmartPtr<BlenderParam> param;
    BlenderParams::iterator i = param_map.find (key.ptr ());
    if (i == param_map.end ()) {
        param = new BlenderParam (idx, NULL, NULL, NULL);
        XCAM_ASSERT (param.ptr ());
        param->stitch_param = key;
        param_map.insert (std::make_pair ((void*)key.ptr (), param));
    } else {
        param = (*i).second;
    }

    return param;
}

XCamReturn
StitcherImpl::start_feature_match (
    const SmartPtr<VideoBuffer> &left_buf,
    const SmartPtr<VideoBuffer> &right_buf,
    const uint32_t idx)
{
#if ENABLE_FEATURE_MATCH
    _overlaps[idx].matcher->reset_offsets ();
    _overlaps[idx].matcher->feature_match (left_buf, right_buf);

    Rect left_ovlap, right_ovlap;
    _overlaps[idx].matcher->get_crop_rect (left_ovlap, right_ovlap);

    float left_offsetx = _overlaps[idx].matcher->get_current_left_offset_x ();
    Factor left_factor, right_factor;

    uint32_t left_idx = idx;
    float center_x = (float) _stitcher->get_center (left_idx).slice_center_x;
    float feature_center_x = (float)left_ovlap.pos_x + (left_ovlap.width / 2.0f);
    float range = feature_center_x - center_x;
    XCAM_ASSERT (range > 1.0f);
    right_factor.x = (range + left_offsetx / 2.0f) / range;
    right_factor.y = 1.0f;
    XCAM_ASSERT (right_factor.x > 0.0f && right_factor.x < 2.0f);

    uint32_t right_idx = (idx + 1) % _stitcher->get_camera_num ();
    center_x = (float) _stitcher->get_center (right_idx).slice_center_x;
    feature_center_x = (float)right_ovlap.pos_x + (right_ovlap.width / 2.0f);
    range = center_x - feature_center_x;
    XCAM_ASSERT (range > 1.0f);
    left_factor.x = (range + left_offsetx / 2.0f) / range;
    left_factor.y = 1.0f;
    XCAM_ASSERT (left_factor.x > 0.0f && left_factor.x < 2.0f);

    {
        SmartLock locker (_map_mutex);
        _fisheye[left_idx].right_match_factor = right_factor;
        _fisheye[right_idx].left_match_factor = left_factor;
    }

    return XCAM_RETURN_NO_ERROR;
#else
    XCAM_LOG_ERROR ("FeatureMatch unsupported");
    return XCAM_RETURN_ERROR_PARAM;
#endif
}

XCamReturn
StitcherImpl::start_overlap_task (uint32_t idx, const SmartPtr<BlenderParam> &param)
{
    if (complete_stitch (_stitcher, param->stitch_param->frame_count)) {
        XCamReturn ret = _overlaps[idx].blender->execute_buffer (param, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft-stitcher:%s blender idx:%d failed", XCAM_STR (_stitcher->get_name ()), idx);
    }

#if ENABLE_FEATURE_MATCH
    if (_stitcher->get_fm_mode () != FMNone) {
        if (_stitcher->get_fm_status () != FMStatusWholeWay &&
                param->stitch_param->frame_count >= _stitcher->get_fm_frames ())
            return XCAM_RETURN_NO_ERROR;

        XCamReturn ret = start_feature_match (param->in_buf, param->in1_buf, idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft-stitcher:%s feature match idx:%d failed", XCAM_STR (_stitcher->get_name ()), idx);
    }
#endif

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_overlap_tasks (
    const SmartPtr<SoftStitcher::StitcherParam> &param,
    const uint32_t idx, const SmartPtr<VideoBuffer> &buf)
{
    SmartPtr<BlenderParam> cur_param, prev_param;
    const uint32_t camera_num = _stitcher->get_camera_num ();
    uint32_t pre_idx = (idx + camera_num - 1) % camera_num;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    {
        SmartPtr<BlenderParam> param_b;

        SmartLock locker (_map_mutex);
        param_b = _overlaps[idx].find_blender_param_in_map (param, idx);
        param_b->in_buf = buf;
        if (param_b->in_buf.ptr () && param_b->in1_buf.ptr ()) {
            cur_param = param_b;
            _overlaps[idx].param_map.erase (param.ptr ());
        }

        param_b = _overlaps[pre_idx].find_blender_param_in_map (param, pre_idx);
        param_b->in1_buf = buf;
        if (param_b->in_buf.ptr () && param_b->in1_buf.ptr ()) {
            prev_param = param_b;
            _overlaps[pre_idx].param_map.erase (param.ptr ());
        }
    }

    if (cur_param.ptr ()) {
        cur_param->out_buf = param->out_buf;
        start_overlap_task (idx, cur_param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft-stitcher:%s start overlap task idx:%d failed", XCAM_STR (_stitcher->get_name ()), idx);

    }

    if (prev_param.ptr ()) {
        prev_param->out_buf = param->out_buf;
        start_overlap_task (pre_idx, prev_param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "soft-stitcher:%s start overlap task idx:%d failed", XCAM_STR (_stitcher->get_name ()), idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
Copier::start_copy_task (
    const SmartPtr<ImageHandler::Parameters> &param,
    const uint32_t idx, const SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (copy_task.ptr ());

    SmartPtr<VideoBuffer> in_buf = buf, out_buf = param->out_buf;
    const VideoBufferInfo &in_info = in_buf->get_video_info ();
    const VideoBufferInfo &out_info = out_buf->get_video_info ();

    SmartPtr<StitcherCopyArgs> args = new StitcherCopyArgs (idx, param);
    args->in_luma = new UcharImage (
        in_buf, copy_area.in_area.width, copy_area.in_area.height, in_info.strides[0],
        in_info.offsets[0] + copy_area.in_area.pos_x + copy_area.in_area.pos_y * in_info.strides[0]);

    args->out_luma = new UcharImage (
        out_buf, copy_area.out_area.width, copy_area.out_area.height, out_info.strides[0],
        out_info.offsets[0] + copy_area.out_area.pos_x + copy_area.out_area.pos_y * out_info.strides[0]);

    if ((V4L2_PIX_FMT_NV12 == in_info.format) && (V4L2_PIX_FMT_NV12 == out_info.format)) {
        args->in_uv = new Uchar2Image (
            in_buf, copy_area.in_area.width / 2, copy_area.in_area.height / 2, in_info.strides[1],
            in_info.offsets[1] + copy_area.in_area.pos_x + copy_area.in_area.pos_y / 2 * in_info.strides[1]);
        args->out_uv = new Uchar2Image (
            out_buf, copy_area.out_area.width / 2, copy_area.out_area.height / 2, out_info.strides[1],
            out_info.offsets[1] + copy_area.out_area.pos_x + copy_area.out_area.pos_y / 2 * out_info.strides[1]);
    } else if ((V4L2_PIX_FMT_YUV420 == in_info.format) && (V4L2_PIX_FMT_YUV420 == out_info.format)) {
        args->in_u = new UcharImage (
            in_buf, copy_area.in_area.width / 2, copy_area.in_area.height / 2, in_info.strides[1],
            in_info.offsets[1] + copy_area.in_area.pos_x / 2 + copy_area.in_area.pos_y / 2 * in_info.strides[1]);
        args->in_v = new UcharImage (
            in_buf, copy_area.in_area.width / 2, copy_area.in_area.height / 2, in_info.strides[2],
            in_info.offsets[2] + copy_area.in_area.pos_x / 2 + copy_area.in_area.pos_y / 2 * in_info.strides[2]);
        args->out_u = new UcharImage (
            out_buf, copy_area.out_area.width / 2, copy_area.out_area.height / 2, out_info.strides[1],
            out_info.offsets[1] + copy_area.out_area.pos_x / 2 + copy_area.out_area.pos_y / 2 * out_info.strides[1]);
        args->out_v = new UcharImage (
            out_buf, copy_area.out_area.width / 2, copy_area.out_area.height / 2, out_info.strides[2],
            out_info.offsets[2] + copy_area.out_area.pos_x / 2 + copy_area.out_area.pos_y / 2 * out_info.strides[2]);
    } else {
        XCAM_LOG_ERROR ("copy_task buffer pixel format:%d unsupported!", in_info.format);
    }

    uint32_t thread_x = 1, thread_y = 4;
    WorkSize global_size (1, xcam_ceil (copy_area.in_area.height, 2) / 2);
    WorkSize local_size (
        xcam_ceil (global_size.value[0], thread_x) / thread_x,
        xcam_ceil (global_size.value[1], thread_y) / thread_y);

    copy_task->set_local_size (local_size);
    copy_task->set_global_size (global_size);

    return copy_task->work (args);
}

XCamReturn
StitcherImpl::start_copy_tasks (
    const SmartPtr<SoftStitcher::StitcherParam> &param,
    const uint32_t idx, const SmartPtr<VideoBuffer> &buf)
{
    uint32_t size = _stitcher->get_copy_area ().size ();
    for (uint32_t i = 0; i < size; ++i) {
        if(_copiers[i].copy_area.in_idx == idx) {
            XCamReturn ret = _copiers[i].start_copy_task (param, idx, buf);
            XCAM_FAIL_RETURN (
                ERROR, xcam_ret_is_ok (ret), ret,
                "soft-stitcher:%s start copy task failed, idx:%d", XCAM_STR (_stitcher->get_name ()), idx);
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::stop ()
{
    uint32_t cam_num = _stitcher->get_camera_num ();
    for (uint32_t i = 0; i < cam_num; ++i) {
        if (_fisheye[i].mapper.ptr ()) {
            _fisheye[i].mapper->terminate ();
            _fisheye[i].mapper.release ();
        }
        if (_fisheye[i].buf_pool.ptr ()) {
            _fisheye[i].buf_pool->stop ();
        }

        if (_overlaps[i].blender.ptr ()) {
            _overlaps[i].blender->terminate ();
            _overlaps[i].blender.release ();
        }
    }

    for (Copiers::iterator i_copy = _copiers.begin (); i_copy != _copiers.end (); ++i_copy) {
        Copier &copy = *i_copy;
        if (copy.copy_task.ptr ()) {
            copy.copy_task->stop ();
            copy.copy_task.release ();
        }
    }

    if (_geomap_pool.ptr ()) {
        _geomap_pool->stop ();
    }

    return XCAM_RETURN_NO_ERROR;
}

};

SoftStitcher::SoftStitcher (const char *name)
    : SoftHandler (name)
    , Stitcher (SOFT_STITCHER_ALIGNMENT_X, SOFT_STITCHER_ALIGNMENT_Y)
{
    SmartPtr<SoftStitcherPriv::StitcherImpl> impl = new SoftStitcherPriv::StitcherImpl (this);
    XCAM_ASSERT (impl.ptr ());
    _impl = impl;

#if ENABLE_FEATURE_MATCH
#ifndef ANDROID
    cv::ocl::setUseOpenCL (false);
#endif
#endif
}

SoftStitcher::~SoftStitcher ()
{
}

XCamReturn
SoftStitcher::stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf)
{
    XCAM_FAIL_RETURN (
        ERROR, !in_bufs.empty (), XCAM_RETURN_ERROR_PARAM,
        "soft-stitcher:%s stitch buffer failed, in_bufs is empty", XCAM_STR (get_name ()));

    uint32_t frame_count = (get_fm_frame_count () == UINT32_MAX) ? 0 : (get_fm_frame_count () + 1);
    set_fm_frame_count (frame_count);

    SmartPtr<StitcherParam> param = new StitcherParam;
    param->out_buf = out_buf;
    param->in_buf_num = in_bufs.size ();
    param->frame_count = frame_count;

    uint32_t count = 0;
    for (VideoBufferList::const_iterator i = in_bufs.begin (); i != in_bufs.end (); ++i) {
        SmartPtr<VideoBuffer> buf = *i;
        XCAM_ASSERT (buf.ptr ());
        param->in_bufs[count++] = buf;
    }

    uint32_t format = param->in_bufs[0]->get_format ();
    _impl->set_pixel_format (format);

    if (in_bufs.size () == 1) {
        for (uint32_t i = 1; i < get_camera_num (); ++i) {
            param->in_bufs[i] = param->in_bufs[0];
        }
    }

    XCamReturn ret = execute_buffer (param, true);

    if (!out_buf.ptr () && xcam_ret_is_ok (ret)) {
        out_buf = param->out_buf;
    }

    return ret;
}

XCamReturn
SoftStitcher::terminate ()
{
    _impl->stop ();
    return SoftHandler::terminate ();
}

XCamReturn
SoftStitcher::start_task_count (const SmartPtr<SoftStitcher::StitcherParam> &param)
{
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (_impl.ptr ());

    SmartLock locker (_impl->_map_mutex);

    XCAM_FAIL_RETURN (
        ERROR, check_work_continue (param, XCAM_RETURN_NO_ERROR), XCAM_RETURN_ERROR_PARAM,
        "soft-stitcher:%s start task count failed in work check", XCAM_STR (get_name ()));

    if (_impl->_task_counts.find (param.ptr ()) != _impl->_task_counts.end ()) {
        XCAM_LOG_ERROR ("tasks already started, this should never happen.");
        return XCAM_RETURN_ERROR_UNKNOWN;
    }

    int32_t count = get_camera_num ();
    if (complete_stitch (this, param->frame_count)) {
        count += get_copy_area ().size ();
    }

    XCAM_LOG_DEBUG ("stitcher :%s start task count :%d", XCAM_STR(get_name ()), count);
    _impl->_task_counts.insert (std::make_pair((void*)param.ptr(), count));

    return XCAM_RETURN_NO_ERROR;
}

void
SoftStitcher::geomap_done (
    const SmartPtr<ImageHandler> &handler,
    const SmartPtr<ImageHandler::Parameters> &base,
    const XCamReturn error)
{
    SmartPtr<SoftStitcherPriv::HandlerParam> geomap_param = base.dynamic_cast_ptr<SoftStitcherPriv::HandlerParam> ();
    XCAM_ASSERT (geomap_param.ptr ());
    SmartPtr<SoftStitcher::StitcherParam> param = geomap_param->stitch_param;
    XCAM_ASSERT (param.ptr ());
    XCAM_UNUSED (handler);

    if (!check_work_continue (param, error))
        return;

    XCAM_LOG_DEBUG ("soft-stitcher:%s camera(idx:%d) geomap done", XCAM_STR (get_name ()), geomap_param->idx);
    stitcher_dump_buf (geomap_param->out_buf, geomap_param->idx, "stitcher-geomap");

    //start both blender and feature match
    XCamReturn ret = _impl->start_overlap_tasks (param, geomap_param->idx, geomap_param->out_buf);
    if (!xcam_ret_is_ok (ret)) {
        work_broken (param, ret);
    }

    if (complete_stitch (this, param->frame_count)) {
        ret = _impl->start_copy_tasks (param, geomap_param->idx, geomap_param->out_buf);
        if (!xcam_ret_is_ok (ret)) {
            work_broken (param, ret);
        }
    } else {
        if (!check_work_continue (param, error)) {
            _impl->remove_task_count (param);
            return;
        }
        if (_impl->dec_task_count (param) == 0) {
            work_well_done (param, error);
        }
    }
}

void
SoftStitcher::blender_done (
    const SmartPtr<ImageHandler> &handler,
    const SmartPtr<ImageHandler::Parameters> &base,
    const XCamReturn error)
{
    SmartPtr<SoftStitcherPriv::BlenderParam> blender_param = base.dynamic_cast_ptr<SoftStitcherPriv::BlenderParam> ();
    XCAM_ASSERT (blender_param.ptr ());
    SmartPtr<SoftStitcher::StitcherParam> param = blender_param->stitch_param;
    XCAM_ASSERT (param.ptr ());
    XCAM_UNUSED (handler);

    if (!check_work_continue (param, error)) {
        _impl->remove_task_count (param);
        return;
    }

    stitcher_dump_buf (blender_param->out_buf, blender_param->idx, "stitcher-blend");
    XCAM_LOG_DEBUG ("blender:(%s) overlap:%d done", XCAM_STR (handler->get_name ()), blender_param->idx);

    if (_impl->dec_task_count (param) == 0) {
        work_well_done (param, error);
    }
}

void
SoftStitcher::copy_task_done (
    const SmartPtr<Worker> &worker,
    const SmartPtr<Worker::Arguments> &base,
    const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (worker.ptr ());
    SmartPtr<SoftStitcherPriv::StitcherCopyArgs> args = base.dynamic_cast_ptr<SoftStitcherPriv::StitcherCopyArgs> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<SoftStitcher::StitcherParam> param =
        args->get_param ().dynamic_cast_ptr<SoftStitcher::StitcherParam> ();
    XCAM_ASSERT (param.ptr ());

    if (!check_work_continue (param, error)) {
        _impl->remove_task_count (param);
        return;
    }
    XCAM_LOG_DEBUG ("soft-stitcher:%s camera(idx:%d) copy done", XCAM_STR (get_name ()), args->idx);

    if (_impl->dec_task_count (param) == 0) {
        work_well_done (param, error);
    }
}

XCamReturn
SoftStitcher::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_UNUSED (param);
    XCAM_ASSERT (_impl.ptr ());

    XCamReturn ret = init_camera_info ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher(%s) init camera info failed", XCAM_STR (get_name ()));

    ret = estimate_round_slices ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher:%s estimate round view slices failed", XCAM_STR (get_name ()));

    ret = estimate_coarse_crops ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher:%s estimate coarse crops failed", XCAM_STR (get_name ()));

    ret = mark_centers ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher:%s mark centers failed", XCAM_STR (get_name ()));

    ret = estimate_overlap ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher:%s estimake coarse overlap failed", XCAM_STR (get_name ()));

    ret = update_copy_areas ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher:%s update copy areas failed", XCAM_STR (get_name ()));

    uint32_t camera_count = get_camera_num ();
    ret = _impl->init_config (camera_count);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher:%s initialize private config failed", XCAM_STR (get_name ()));

    ret = _impl->gen_geomap_table ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "soft-stitcher:%s gen_geomap_table failed", XCAM_STR (get_name ()));

    VideoBufferInfo out_info;
    uint32_t out_width, out_height;
    get_output_size (out_width, out_height);
    XCAM_FAIL_RETURN (
        ERROR, out_width && out_height, XCAM_RETURN_ERROR_PARAM,
        "soft-stitcher:%s output size was not set", XCAM_STR(get_name ()));

    out_info.init (
        _impl->get_pixel_format (), out_width, out_height,
        XCAM_ALIGN_UP (out_width, SOFT_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (out_height, SOFT_STITCHER_ALIGNMENT_Y));
    set_out_video_info (out_info);

    return ret;
}

XCamReturn
SoftStitcher::start_work (const SmartPtr<Parameters> &base)
{
    SmartPtr<StitcherParam> param = base.dynamic_cast_ptr<StitcherParam> ();

    XCAM_FAIL_RETURN (
        ERROR, param.ptr () && param->in_buf_num > 0 && param->in_bufs[0].ptr (), XCAM_RETURN_ERROR_PARAM,
        "soft_stitcher:%s start_work failed, params(in_buf_num) in_bufs are set",
        XCAM_STR (get_name ()));

    XCamReturn ret = start_task_count (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), XCAM_RETURN_ERROR_PARAM,
        "soft_stitcher:%s start blender count failed", XCAM_STR (get_name ()));

    ret = _impl->start_geomap_works (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), XCAM_RETURN_ERROR_PARAM,
        "soft_stitcher:%s start geomap works failed", XCAM_STR (get_name ()));

    //for (uint32_t i = 0; i < param->in_buf_num; ++i) {
    //    param->in_bufs[i].release ();
    //}

    return ret;
}

SmartPtr<Stitcher>
Stitcher::create_soft_stitcher ()
{
    return new SoftStitcher;
}

}

