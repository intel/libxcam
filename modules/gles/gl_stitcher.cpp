/*
 * gl_stitcher.cpp - GL stitcher implementation
 *
 *  Copyright (c) 2018 Intel Corporation
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
 */

#include "fisheye_dewarp.h"
#include "gl_blender.h"
#include "gl_copy_handler.h"
#include "gl_geomap_handler.h"
#include "gl_stitcher.h"
#include "gl_sync.h"
#include "gl_video_buffer.h"
#include "interface/feature_match.h"

#define GL_STITCHER_ALIGNMENT_X 16
#define GL_STITCHER_ALIGNMENT_Y 4

#define MAP_FACTOR_X  16
#define MAP_FACTOR_Y  16

#define DUMP_BUFFER 0

namespace XCam {

#if DUMP_BUFFER
static void
dump_buf (const SmartPtr<VideoBuffer> &buf, uint32_t idx, const char *prefix)
{
    char name[256];
    snprintf (name, 256, "%s-%d", prefix, idx);
    dump_buf_perfix_path (buf, name);
}
#endif

namespace GLStitcherPriv {

struct Factor {
    float x, y;

    Factor () : x (1.0f), y (1.0f) {}
    void reset () {
        x = 1.0f;
        y = 1.0f;
    }
};

typedef std::vector<SmartPtr<GLCopyHandler>> Copiers;

class StitcherImpl {
    friend class XCam::GLStitcher;

public:
    explicit StitcherImpl (GLStitcher *handler);

    XCamReturn init_config ();

    XCamReturn start_geomappers (const SmartPtr<GLStitcher::StitcherParam> &param);
    XCamReturn start_blenders (const SmartPtr<GLStitcher::StitcherParam> &param);
    XCamReturn start_copiers (const SmartPtr<GLStitcher::StitcherParam> &param);
    XCamReturn start_feature_matches ();

    XCamReturn stop ();

private:
    SmartPtr<GLGeoMapHandler> create_geomapper ();

    XCamReturn init_geomapper (uint32_t idx);
    XCamReturn init_blender (uint32_t idx);
    XCamReturn init_copier (Stitcher::CopyArea &area);

    XCamReturn gen_geomap_table (const Stitcher::RoundViewSlice &view_slice, uint32_t cam_idx);
    bool update_geomapper_factors (uint32_t idx);
    void calc_geomap_factors (
        uint32_t idx,
        const Factor &last_left_factor, const Factor &last_right_factor,
        Factor &cur_left, Factor &cur_right);
    bool get_and_reset_fm_factors (uint32_t idx, Factor &left, Factor &right);

#if HAVE_OPENCV
    XCamReturn init_feature_match (uint32_t idx);
    XCamReturn create_feature_match (SmartPtr<FeatureMatch> &matcher);
    XCamReturn start_feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf, uint32_t idx);
#endif

private:
    StitchInfo                    _stitch_info;
    uint32_t                      _camera_num;
    FisheyeDewarpMode             _dewarp_mode;

    SmartPtr<GLGeoMapHandler>     _geomapper[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<FeatureMatch>        _matcher[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<GLBlender>           _blender[XCAM_STITCH_MAX_CAMERAS];
    Copiers                       _copiers;

    SmartPtr<BufferPool>          _geomap_pool[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<VideoBuffer>         _geomap_buf[XCAM_STITCH_MAX_CAMERAS];

    FisheyeInfo                   _fisheye_info[XCAM_STITCH_MAX_CAMERAS];
    Factor                        _left_fm_factor[XCAM_STITCH_MAX_CAMERAS];
    Factor                        _right_fm_factor[XCAM_STITCH_MAX_CAMERAS];

    GLStitcher                   *_stitcher;
};

StitcherImpl::StitcherImpl (GLStitcher *handler)
    : _camera_num (0)
    , _dewarp_mode (DewarpSphere)
    , _stitcher (handler)
{
}

XCamReturn
StitcherImpl::init_config ()
{
    _camera_num = _stitcher->get_camera_num ();
    _dewarp_mode = _stitcher->get_dewarp_mode ();
    if (_dewarp_mode == DewarpSphere)
        _stitch_info = _stitcher->get_stitch_info ();

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        ret = init_geomapper (idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher init geo mapper failed, idx: %d", idx);

#if HAVE_OPENCV
        ret = init_feature_match (idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher init feature match failed, idx: %d", idx);
#endif

        init_blender (idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher init blender failed, idx: %d", idx);
    }

    Stitcher::CopyAreaArray areas = _stitcher->get_copy_area ();
    uint32_t size = areas.size ();
    for (uint32_t idx = 0; idx < size; ++idx) {
        XCAM_ASSERT (areas[idx].in_idx < size);

        ret = init_copier (areas[idx]);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher init copier failed, idx: %d", areas[idx].in_idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_geomappers (const SmartPtr<GLStitcher::StitcherParam> &param)
{
    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        _geomap_buf[idx] = _geomap_pool[idx]->get_buffer ();
        SmartPtr<ImageHandler::Parameters> geomap_params = new ImageHandler::Parameters ();
        geomap_params->in_buf = param->in_bufs[idx];
        geomap_params->out_buf = _geomap_buf[idx];

        update_geomapper_factors (idx);

        XCamReturn ret = _geomapper[idx]->execute_buffer (geomap_params, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher execute geomapper failed, idx: %d", idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_blenders (const SmartPtr<GLStitcher::StitcherParam> &param)
{
    for (uint32_t idx = 0; idx <_camera_num; ++idx) {
        uint32_t next_idx = (idx + 1) % _camera_num;

        SmartPtr<GLBlender::BlenderParam> blend_param =
            new GLBlender::BlenderParam (_geomap_buf[idx], _geomap_buf[next_idx], param->out_buf);

        XCamReturn ret = _blender[idx]->execute_buffer (blend_param, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher execute blender failed, idx: %d", idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_copiers (const SmartPtr<GLStitcher::StitcherParam> &param)
{
    for (uint32_t idx = 0; idx < _copiers.size (); ++idx) {
        uint32_t in_idx = _copiers[idx]->get_index ();

        SmartPtr<ImageHandler::Parameters> copy_params = new ImageHandler::Parameters ();
        copy_params->in_buf = _geomap_buf[in_idx];
        copy_params->out_buf = param->out_buf;

        XCamReturn ret = _copiers[idx]->execute_buffer (copy_params, false);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher execute copier failed, idx: %d, in_idx: %d", idx, in_idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

#if HAVE_OPENCV
XCamReturn
StitcherImpl::start_feature_match (
    const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf, uint32_t idx)
{
    _matcher[idx]->reset_offsets ();
    _matcher[idx]->feature_match (left_buf, right_buf);

    Rect left_ovlap, right_ovlap;
    _matcher[idx]->get_crop_rect (left_ovlap, right_ovlap);

    float left_offsetx = _matcher[idx]->get_current_left_offset_x ();
    Factor left_factor, right_factor;

    uint32_t left_idx = idx;
    float center_x = (float) _stitcher->get_center (left_idx).slice_center_x;
    float feature_center_x = (float)left_ovlap.pos_x + (left_ovlap.width / 2.0f);
    float range = feature_center_x - center_x;
    XCAM_ASSERT (range > 1.0f);
    right_factor.x = (range + left_offsetx / 2.0f) / range;
    right_factor.y = 1.0f;
    XCAM_ASSERT (right_factor.x > 0.0f && right_factor.x < 2.0f);

    uint32_t right_idx = (idx + 1) % _camera_num;
    center_x = (float) _stitcher->get_center (right_idx).slice_center_x;
    feature_center_x = (float)right_ovlap.pos_x + (right_ovlap.width / 2.0f);
    range = center_x - feature_center_x;
    XCAM_ASSERT (range > 1.0f);
    left_factor.x = (range + left_offsetx / 2.0f) / range;
    left_factor.y = 1.0f;
    XCAM_ASSERT (left_factor.x > 0.0f && left_factor.x < 2.0f);

    _right_fm_factor[left_idx] = right_factor;
    _left_fm_factor[right_idx] = left_factor;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_feature_matches ()
{
    for (uint32_t idx = 0; idx <_camera_num; ++idx) {
        uint32_t next_idx = (idx + 1) % _camera_num;

        XCamReturn ret = start_feature_match (_geomap_buf[idx], _geomap_buf[next_idx], idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher execute feature match failed, idx: %d", idx);
    }

    return XCAM_RETURN_NO_ERROR;
}
#endif

XCamReturn
StitcherImpl::stop ()
{
    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        if (_geomapper[idx].ptr ()) {
            _geomapper[idx]->terminate ();
            _geomapper[idx].release ();
        }
        if (_matcher[idx].ptr ()) {
            _matcher[idx].release ();
        }
        if (_blender[idx].ptr ()) {
            _blender[idx].release ();
        }
        if (_geomap_pool[idx].ptr ()) {
            _geomap_pool[idx]->stop ();
        }
    }

    for (Copiers::iterator i = _copiers.begin (); i != _copiers.end (); ++i) {
        SmartPtr<GLCopyHandler> &copier = *i;
        if (copier.ptr ()) {
            copier->terminate ();
            copier.release ();
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<GLGeoMapHandler>
StitcherImpl::create_geomapper ()
{
    SmartPtr<GLGeoMapHandler> mapper;
    GeoMapScaleMode scale_mode = _stitcher->get_scale_mode ();
    if (scale_mode == ScaleSingleConst)
        mapper = new GLGeoMapHandler ("stitcher_singleconst_remapper");
    else if (scale_mode == ScaleDualConst) {
        mapper = new GLDualConstGeoMapHandler ("stitcher_dualconst_remapper");
    } else {
        XCAM_LOG_ERROR ("gl-stitcher unsupported geomap scale mode: %d", scale_mode);
    }
    XCAM_ASSERT (mapper.ptr ());

    return mapper;
}

XCamReturn
StitcherImpl::init_geomapper (uint32_t idx)
{
    if (_dewarp_mode == DewarpSphere)
        _fisheye_info[idx] = _stitch_info.fisheye_info[idx];

    _geomapper[idx] = create_geomapper ();

    const Stitcher::RoundViewSlice &slice = _stitcher->get_round_view_slice (idx);
    _geomapper[idx]->set_output_size (slice.width, slice.height);

    XCamReturn ret = gen_geomap_table (slice, idx);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher generate geomap table failed, idx: %d", idx);

    VideoBufferInfo info;
    info.init (
        V4L2_PIX_FMT_NV12, slice.width, slice.height,
        XCAM_ALIGN_UP (slice.width, GL_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (slice.height, GL_STITCHER_ALIGNMENT_Y));

    SmartPtr<BufferPool> pool = new GLVideoBufferPool (info);
    XCAM_ASSERT (pool.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, pool->reserve (XCAM_GL_RESERVED_BUF_COUNT), XCAM_RETURN_ERROR_MEM,
        "gl-stitcher reserve geomap buffer pool failed, buffer size: %dx%d",
        info.width, info.height);
    _geomap_pool[idx] = pool;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_blender (uint32_t idx)
{
    SmartPtr<GLBlender> &blender = _blender[idx];
    blender = create_gl_blender ().dynamic_cast_ptr<GLBlender>();
    XCAM_ASSERT (blender.ptr ());

    uint32_t width, height;
    _stitcher->get_output_size (width, height);
    blender->set_output_size (width, height);
    blender->set_pyr_levels (_stitcher->get_blend_pyr_levels ());

    const Stitcher::ImageOverlapInfo &overlap_info = _stitcher->get_overlap (idx);
    Stitcher::ImageOverlapInfo overlap = overlap_info;
    if (_dewarp_mode == DewarpSphere && _stitch_info.merge_width[idx] > 0) {
        uint32_t specific_merge_width = _stitch_info.merge_width[idx];
        XCAM_ASSERT (uint32_t (overlap.left.width) >= specific_merge_width);

        uint32_t ext_width = (overlap.left.width - specific_merge_width) / 2;
        ext_width = XCAM_ALIGN_UP (ext_width, GL_STITCHER_ALIGNMENT_X);

        overlap.left.pos_x += ext_width;
        overlap.left.width = specific_merge_width;
        overlap.right.pos_x += ext_width;
        overlap.right.width = specific_merge_width;
        overlap.out_area.pos_x += ext_width;
        overlap.out_area.width = specific_merge_width;
    }

    blender->set_merge_window (overlap.out_area);
    blender->set_input_valid_area (overlap.left, 0);
    blender->set_input_valid_area (overlap.right, 1);
    blender->set_input_merge_area (overlap.left, 0);
    blender->set_input_merge_area (overlap.right, 1);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_copier (Stitcher::CopyArea &area)
{
    if (_dewarp_mode == DewarpSphere && _stitch_info.merge_width[area.in_idx] > 0) {
        uint32_t specific_merge_width = _stitch_info.merge_width[area.in_idx];
        const Stitcher::ImageOverlapInfo overlap_info = _stitcher->get_overlap (area.in_idx);
        XCAM_ASSERT (uint32_t (overlap_info.left.width) >= specific_merge_width);

        uint32_t ext_width = (overlap_info.left.width - specific_merge_width) / 2;
        ext_width = XCAM_ALIGN_UP (ext_width, GL_STITCHER_ALIGNMENT_X);

        area.in_area.width = (area.in_idx == 0) ?
            (area.in_area.width + ext_width) : (area.in_area.width + ext_width * 2);
        area.out_area.width = area.in_area.width;
        if (area.out_area.pos_x > 0) {
            area.in_area.pos_x -= ext_width;
            area.out_area.pos_x -= ext_width;
        }
    }

    SmartPtr<GLCopyHandler> copier = new GLCopyHandler ("stitch_copy");
    XCAM_ASSERT (copier.ptr ());

    copier->enable_allocator (false);
    copier->set_copy_area (area.in_idx, area.in_area, area.out_area);
    _copiers.push_back (copier);

    XCAM_LOG_DEBUG (
        "gl-stitcher copy area idx: %d, input area: %d, %d, %d, %d, output area: %d, %d, %d, %d",
        area.in_idx,
        area.in_area.pos_x, area.in_area.pos_y, area.in_area.width, area.in_area.height,
        area.out_area.pos_x, area.out_area.pos_y, area.out_area.width, area.out_area.height);

    return XCAM_RETURN_NO_ERROR;
}

#if HAVE_OPENCV
XCamReturn
StitcherImpl::create_feature_match (SmartPtr<FeatureMatch> &matcher)
{
#ifndef ANDROID
    FeatureMatchMode fm_mode = _stitcher->get_fm_mode ();
    switch (fm_mode) {
    case FMNone:
        return XCAM_RETURN_NO_ERROR;
    case FMDefault:
        matcher = FeatureMatch::create_default_feature_match ();
        break;
    case FMCluster:
        matcher = FeatureMatch::create_cluster_feature_match ();
        break;
    case FMCapi:
        matcher = FeatureMatch::create_capi_feature_match ();
        break;
    default:
        XCAM_LOG_ERROR ("gl-stitcher unsupported feature match mode: %d", fm_mode);
        return XCAM_RETURN_ERROR_PARAM;
    }
#else
    matcher = FeatureMatch::create_capi_feature_match ();
#endif
    XCAM_ASSERT (matcher.ptr ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_feature_match (uint32_t idx)
{
    if (_stitcher->get_fm_mode () == FMNone)
        return XCAM_RETURN_NO_ERROR;

    SmartPtr<FeatureMatch> &matcher = _matcher[idx];
    XCamReturn ret = create_feature_match (matcher);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher init feature match failed, idx: %d", idx);

    matcher->set_config (_stitcher->get_fm_config ());
    matcher->set_fm_index (idx);

    const BowlDataConfig &bowl = _stitcher->get_bowl_config ();
    const Stitcher::ImageOverlapInfo &info = _stitcher->get_overlap (idx);
    Rect left_ovlap = info.left;
    Rect right_ovlap = info.right;

    if (_dewarp_mode == DewarpSphere) {
        const FMRegionRatio &ratio = _stitcher->get_fm_region_ratio ();

        left_ovlap.pos_y = left_ovlap.height * ratio.pos_y;
        left_ovlap.height = left_ovlap.height * ratio.height;
        right_ovlap.pos_y = left_ovlap.pos_y;
        right_ovlap.height = left_ovlap.height;
    } else {
        left_ovlap.pos_y = 0;
        left_ovlap.height =
            int32_t (bowl.wall_height / (bowl.wall_height + bowl.ground_length) * left_ovlap.height);
        right_ovlap.pos_y = 0;
        right_ovlap.height = left_ovlap.height;
    }
    matcher->set_crop_rect (left_ovlap, right_ovlap);

    return XCAM_RETURN_NO_ERROR;
}
#endif

XCamReturn
StitcherImpl::gen_geomap_table (
    const Stitcher::RoundViewSlice &view_slice, uint32_t cam_idx)
{
    SmartPtr<FisheyeDewarp> dewarper;
    if(_dewarp_mode == DewarpBowl) {
        BowlDataConfig bowl = _stitcher->get_bowl_config ();
        bowl.angle_start = view_slice.hori_angle_start;
        bowl.angle_end = format_angle (view_slice.hori_angle_start + view_slice.hori_angle_range);
        if (bowl.angle_end < bowl.angle_start)
            bowl.angle_start -= 360.0f;

        XCAM_LOG_DEBUG (
            "gl-stitcher camera idx: %d, info angle start: %.2f, range: %.2f, bowlinfo angle start: %.2f, end: %.2f",
            cam_idx, view_slice.hori_angle_start, view_slice.hori_angle_range, bowl.angle_start, bowl.angle_end);

        CameraInfo cam_info;
        _stitcher->get_camera_info (cam_idx, cam_info);

        SmartPtr<PolyBowlFisheyeDewarp> fd = new PolyBowlFisheyeDewarp ();
        fd->set_intr_param (cam_info.calibration.intrinsic);
        fd->set_extr_param (cam_info.calibration.extrinsic);
        fd->set_bowl_config (bowl);
        dewarper = fd;
    } else {
        FisheyeInfo &info = _fisheye_info[cam_idx];
        float max_dst_latitude = (info.intrinsic.fov > 180.0f) ? 180.0f : info.intrinsic.fov;
        float max_dst_longitude = max_dst_latitude * view_slice.width / view_slice.height;

        SmartPtr<SphereFisheyeDewarp> fd = new SphereFisheyeDewarp ();
        fd->set_fisheye_info (info);
        fd->set_dst_range (max_dst_longitude, max_dst_latitude);
        dewarper = fd;
    }
    XCAM_FAIL_RETURN (
        ERROR, dewarper.ptr (), XCAM_RETURN_ERROR_MEM, "gl-stitcher dewarper is NULL");

    dewarper->set_out_size (view_slice.width, view_slice.height);

    uint32_t table_width = view_slice.width / MAP_FACTOR_X;
    table_width = XCAM_ALIGN_UP (table_width, 4);
    uint32_t table_height = view_slice.height / MAP_FACTOR_Y;
    table_height = XCAM_ALIGN_UP (table_height, 2);
    dewarper->set_table_size (table_width, table_height);

    FisheyeDewarp::MapTable map_table (table_width * table_height);
    dewarper->gen_table (map_table);

    XCAM_FAIL_RETURN (
        ERROR,
        _geomapper[cam_idx]->set_lookup_table (map_table.data (), table_width, table_height),
        XCAM_RETURN_ERROR_UNKNOWN,
        "gl-stitcher set geomap lookup table failed");

    return XCAM_RETURN_NO_ERROR;
}

bool
StitcherImpl::update_geomapper_factors (uint32_t idx)
{
    XCAM_FAIL_RETURN (
        ERROR, _geomapper[idx].ptr (), false, "gl-stitcher geomap handler is empty");

    Factor last_left_factor, last_right_factor, cur_left, cur_right;
    SmartPtr<GLGeoMapHandler> &mapper = _geomapper[idx];
    mapper->get_left_factors (last_left_factor.x, last_left_factor.y);
    mapper->get_right_factors (last_right_factor.x, last_right_factor.y);

    if (XCAM_DOUBLE_EQUAL_AROUND (last_left_factor.x, 0.0f) ||
            XCAM_DOUBLE_EQUAL_AROUND (last_left_factor.y, 0.0f) ||
            XCAM_DOUBLE_EQUAL_AROUND (last_right_factor.x, 0.0f) ||
            XCAM_DOUBLE_EQUAL_AROUND (last_right_factor.y, 0.0f)) { // not started
        return true;
    }

    calc_geomap_factors (idx, last_left_factor, last_right_factor, cur_left, cur_right);
    mapper->update_factors (cur_left.x, cur_left.y, cur_right.x, cur_right.y);

    return true;
}

void
StitcherImpl::calc_geomap_factors (
    uint32_t idx, const Factor &last_left_factor, const Factor &last_right_factor,
    Factor &cur_left, Factor &cur_right)
{
    Factor match_left_factor, match_right_factor;
    get_and_reset_fm_factors (idx, match_left_factor, match_right_factor);

    cur_left.x = last_left_factor.x * match_left_factor.x;
    cur_left.y = last_left_factor.y * match_left_factor.y;
    cur_right.x = last_right_factor.x * match_right_factor.x;
    cur_right.y = last_right_factor.y * match_right_factor.y;
}

bool
StitcherImpl::get_and_reset_fm_factors (uint32_t idx, Factor &left, Factor &right)
{
    XCAM_FAIL_RETURN (
        ERROR, idx < _camera_num, false,
        "gl-stitcher invalid camera index: %d, but camera number: %d", idx, _camera_num);

    left = _left_fm_factor[idx];
    right = _right_fm_factor[idx];

    _left_fm_factor[idx].reset ();
    _right_fm_factor[idx].reset ();

    return true;
}

}

GLStitcher::GLStitcher (const char *name)
    : GLImageHandler (name)
    , Stitcher (GL_STITCHER_ALIGNMENT_X, GL_STITCHER_ALIGNMENT_X)
{
    SmartPtr<GLStitcherPriv::StitcherImpl> impl = new GLStitcherPriv::StitcherImpl (this);
    XCAM_ASSERT (impl.ptr ());
    _impl = impl;
}

GLStitcher::~GLStitcher ()
{
}

XCamReturn
GLStitcher::terminate ()
{
    _impl->stop ();
    return GLImageHandler::terminate ();
}

XCamReturn
GLStitcher::stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf)
{
    XCAM_FAIL_RETURN (
        ERROR, !in_bufs.empty (), XCAM_RETURN_ERROR_PARAM,
        "gl-stitcher stitch buffer failed, input buffers is empty");

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

static XCamReturn
set_output_info (const SmartPtr<GLStitcher> &stitch)
{
    VideoBufferInfo info;
    uint32_t width, height;
    stitch->get_output_size (width, height);
    XCAM_FAIL_RETURN (
        ERROR, width && height, XCAM_RETURN_ERROR_PARAM,
        "gl-stitcher invalid output size %dx%d", width, height);

    info.init (
        V4L2_PIX_FMT_NV12, width, height,
        XCAM_ALIGN_UP (width, GL_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (height, GL_STITCHER_ALIGNMENT_Y));
    stitch->set_out_video_info (info);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLStitcher::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_UNUSED (param);
    XCAM_ASSERT (_impl.ptr ());

    XCamReturn ret = init_camera_info ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher init camera info failed");

    ret = estimate_round_slices ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher estimate round view slices failed");

    ret = estimate_coarse_crops ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher estimate coarse crops failed");

    ret = mark_centers ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher mark centers failed");

    ret = estimate_overlap ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher estimake coarse overlap failed");

    ret = update_copy_areas ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher update copy areas failed");

    ret = _impl->init_config ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher initialize private config failed");

    ret = set_output_info (this);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher set output info failed");

    return ret;
}

static inline bool
continue_stitch (Stitcher *stitcher, uint32_t frame_count)
{
    return (
        stitcher->get_fm_mode () == FMNone ||
        stitcher->get_fm_status () != FMStatusFMFirst ||
        frame_count >= stitcher->get_fm_frames ());
}

static inline bool
need_feature_match (Stitcher *stitcher, uint32_t frame_count)
{
    if (stitcher->get_fm_mode () == FMNone)
        return false;

    if (stitcher->get_fm_status () != FMStatusWholeWay && frame_count >= stitcher->get_fm_frames ())
        return false;

    return true;
}

XCamReturn
GLStitcher::start_work (const SmartPtr<Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());

    SmartPtr<StitcherParam> param = base.dynamic_cast_ptr<StitcherParam> ();
    XCAM_FAIL_RETURN (
        ERROR, param.ptr () && param->in_buf_num > 0 && param->in_bufs[0].ptr (),
        XCAM_RETURN_ERROR_MEM,
        "gl-stitcher execute failed, invalid parameters");

    XCamReturn ret = _impl->start_geomappers (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl_stitcher execute geomappers failed");

    if (continue_stitch (this, param->frame_count)) {
        ret = _impl->start_blenders (param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl_stitcher execute blenders failed");

        ret = _impl->start_copiers (param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl_stitcher execute copiers failed");
    }

    GLSync::flush ();

#if HAVE_OPENCV
    if (need_feature_match (this, param->frame_count)) {
        ret = _impl->start_feature_matches ();
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl_stitcher execute feature matches failed");
    }
#endif

#if DUMP_BUFFER
    GLSync::finish ();
    for (uint32_t idx = 0; idx < get_camera_num (); ++idx) {
        dump_buf (_impl->_geomap_buf[idx], idx, "geomap");
    }
#endif

    return ret;
}

SmartPtr<Stitcher>
Stitcher::create_gl_stitcher ()
{
    return new GLStitcher;
}

}
