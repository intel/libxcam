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
#include "gl_fastmap_blender.h"
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

#define XCAM_FISHEYE_IMG_ROI_RADIUS 0

namespace XCam {

enum GeoMapIdx {
    Copy0 = 0,
    Copy1,
    BlendLeft,
    BlendRight,
    FMLeft,
    FMRight,
    MapMax
};

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

class StitcherImpl {
    friend class XCam::GLStitcher;

public:
    explicit StitcherImpl (GLStitcher *handler);

    XCamReturn init_config ();
    XCamReturn start_geomappers (const SmartPtr<GLStitcher::StitcherParam> &param);
    XCamReturn start_blenders (const SmartPtr<GLStitcher::StitcherParam> &param);
    XCamReturn start_feature_matches ();

    XCamReturn stop ();

private:
    XCamReturn init_geomappers (uint32_t idx);
    XCamReturn init_blender (uint32_t idx);

    XCamReturn init_geomapper (
        SmartPtr<GLGeoMapHandler> &mapper, const SmartPtr<GLBuffer> &lut, const Stitcher::RoundViewSlice &slice);

    XCamReturn gen_geomap_table (
        const Stitcher::RoundViewSlice &view_slice, uint32_t cam_idx, FisheyeDewarp::MapTable &map_table,
        uint32_t table_width, uint32_t table_height);
    bool update_geomapper_factors (const SmartPtr<GLGeoMapHandler> &mapper, uint32_t idx);

    XCamReturn config_geomappers_from_copy ();
    XCamReturn config_geomapper_from_blend (
        const Stitcher::ImageOverlapInfo &overlap, uint32_t idx);
    XCamReturn config_geomapper_from_fm (
        const Stitcher::ImageOverlapInfo &overlap, uint32_t idx, GeoMapIdx fm_idx);

    XCamReturn activate_fastmap (const SmartPtr<GLStitcher::StitcherParam> &param);
    XCamReturn calc_fisheye_img_roi_radius (uint32_t idx);

    XCamReturn release_geomapper_rsc (uint32_t cam_id, GeoMapIdx idx);
    XCamReturn release_unused_rsc ();

#if HAVE_OPENCV
    XCamReturn init_feature_match (uint32_t idx);
    XCamReturn create_feature_match (SmartPtr<FeatureMatch> &matcher);
    XCamReturn start_feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf, uint32_t idx);
    XCamReturn start_fm_geomapper (
        const SmartPtr<GLStitcher::StitcherParam> &param, uint32_t idx, GeoMapIdx fm_idx);
#endif

private:
    StitchInfo                    _stitch_info;
    uint32_t                      _camera_num;
    FisheyeDewarpMode             _dewarp_mode;

    SmartPtr<GLGeoMapHandler>     _geomapper[XCAM_STITCH_MAX_CAMERAS][MapMax];
    SmartPtr<FeatureMatch>        _matcher[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<GLBlender>           _blender[XCAM_STITCH_MAX_CAMERAS];
    SmartPtr<GLFastmapBlender>    _fastmap_blender[XCAM_STITCH_MAX_CAMERAS];

    SmartPtr<BufferPool>          _geomap_pool[XCAM_STITCH_MAX_CAMERAS][MapMax];
    SmartPtr<VideoBuffer>         _geomap_buf[XCAM_STITCH_MAX_CAMERAS][MapMax];

    FisheyeInfo                   _fisheye_info[XCAM_STITCH_MAX_CAMERAS];
    Factor                        _fm_left_factor[XCAM_STITCH_MAX_CAMERAS];
    Factor                        _fm_right_factor[XCAM_STITCH_MAX_CAMERAS];

    bool                          _fastmap_activated;
    bool                          _fastmap_blend_activated;

    uint32_t                      _fisheye_img_roi_radius[XCAM_STITCH_MAX_CAMERAS];

    GLStitcher                   *_stitcher;
};

StitcherImpl::StitcherImpl (GLStitcher *handler)
    : _camera_num (0)
    , _dewarp_mode (DewarpSphere)
    , _fastmap_activated (false)
    , _fastmap_blend_activated (false)
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
        ret = init_geomappers (idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher init geo mappers failed, idx: %d", idx);
    }

    ret = config_geomappers_from_copy ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher config mappers from copy failed");

    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        ret = init_blender (idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher init blender failed, idx: %d", idx);
    }

#if HAVE_OPENCV
    if (_stitcher->need_feature_match ()) {
        for (uint32_t idx = 0; idx < _camera_num; ++idx) {
            ret = init_feature_match (idx);
            XCAM_FAIL_RETURN (
                ERROR, xcam_ret_is_ok (ret), ret,
                "gl-stitcher init feature match failed, idx: %d", idx);
        }
    }
#endif

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
start_geomapper (
    const SmartPtr<GLGeoMapHandler> &geomapper,
    const SmartPtr<VideoBuffer> &in, const SmartPtr<VideoBuffer> &out)
{
    SmartPtr<ImageHandler::Parameters> geomap_params = new ImageHandler::Parameters (in, out);

    XCamReturn ret = geomapper->execute_buffer (geomap_params, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret, "gl-stitcher execute geomapper failed");

    return XCAM_RETURN_NO_ERROR;
}

#if XCAM_FISHEYE_IMG_ROI_RADIUS
XCamReturn
StitcherImpl::calc_fisheye_img_roi_radius (uint32_t idx)
{
    float cx, cy;
    if(_dewarp_mode == DewarpBowl) {
        CameraInfo cam_info;
        _stitcher->get_camera_info (idx, cam_info);

        cx = cam_info.calibration.intrinsic.cx;
        cy = cam_info.calibration.intrinsic.cy;
    } else {
        cx = _stitch_info.fisheye_info[idx].intrinsic.cx;
        cy = _stitch_info.fisheye_info[idx].intrinsic.cy;
    }

    const SmartPtr<GLBuffer> &coordx = _geomapper[idx][Copy0]->get_coordx_buf ();
    const SmartPtr<GLBuffer> &coordy = _geomapper[idx][Copy0]->get_coordy_buf ();
    const GLBufferDesc &desc = coordx->get_buffer_desc ();

    float *xptr = (float *) coordx->map_range (0, desc.size, GL_MAP_READ_BIT);
    float *yptr = (float *) coordy->map_range (0, desc.size, GL_MAP_READ_BIT);
    XCAM_FAIL_RETURN (ERROR, xptr || yptr, XCAM_RETURN_ERROR_MEM, "gl-stitcher map range failed");

    uint32_t i;
    float max_r = 0.0f, x, y, r;

    for (uint32_t h = 0; h < desc.height; ++h) {
        for (uint32_t w = 0; w < desc.width; ++w) {
            if (h > 0 && h < (desc.height - 1) && w > 0 && w < (desc.width - 1))
                break;

            i = h * desc.width + w;
            x = fabs (xptr[i] - cx) + 0.5f;
            y = fabs (yptr[i] - cy) + 0.5f;

            r = sqrt (x * x + y * y);
            max_r = max_r < r ? r : max_r;
        }
    }
    coordx->unmap ();
    coordy->unmap ();

    _fisheye_img_roi_radius[idx] = max_r + 1.0f;
    XCAM_LOG_INFO (
        "calculate fisheye img roi radius fisheye_img_roi_radius = %d",
        _fisheye_img_roi_radius[idx]);

    return XCAM_RETURN_NO_ERROR;
}
#endif

XCamReturn
StitcherImpl::activate_fastmap (const SmartPtr<GLStitcher::StitcherParam> &param)
{
    if (_fastmap_activated)
        return XCAM_RETURN_NO_ERROR;

    _geomapper[0][Copy1]->activate_fastmap ();
    start_geomapper (_geomapper[0][Copy1], param->in_bufs[0], param->out_buf);

    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        _geomapper[idx][Copy0]->activate_fastmap ();
        start_geomapper (_geomapper[idx][Copy0], param->in_bufs[idx], param->out_buf);

        _geomapper[idx][BlendLeft]->activate_fastmap ();
        _geomap_buf[idx][BlendLeft] = _geomap_pool[idx][BlendLeft]->get_buffer ();
        start_geomapper (_geomapper[idx][BlendLeft], param->in_bufs[idx], _geomap_buf[idx][BlendLeft]);

        _geomapper[idx][BlendRight]->activate_fastmap ();
        _geomap_buf[idx][BlendRight] = _geomap_pool[idx][BlendRight]->get_buffer ();
        start_geomapper (_geomapper[idx][BlendRight], param->in_bufs[idx], _geomap_buf[idx][BlendRight]);

#if XCAM_FISHEYE_IMG_ROI_RADIUS
        calc_fisheye_img_roi_radius (idx);
#endif
    }

    if (_stitcher->get_blend_pyr_levels () == 1) {
        _fastmap_blend_activated = true;

        for (uint32_t idx = 0; idx < _camera_num; ++idx) {
            uint32_t next_idx = (idx + _camera_num + 1) % _camera_num;
            SmartPtr<GLFastmapBlender> fastmap_blender = new GLFastmapBlender ("stitcher_fastmap_blender");
            XCAM_ASSERT (fastmap_blender.ptr ());

            fastmap_blender->set_fastmappers (_geomapper[idx][BlendRight], _geomapper[next_idx][BlendLeft]);
            fastmap_blender->set_blender (_blender[idx]);
            fastmap_blender->enable_allocator (false);

            _fastmap_blender[idx] = fastmap_blender;
        }
    }

    _fastmap_activated = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_geomappers (const SmartPtr<GLStitcher::StitcherParam> &param)
{
    if (!_stitcher->need_feature_match ()) {
        activate_fastmap (param);
    }

    if (_stitcher->complete_stitch ()) {
        start_geomapper (_geomapper[0][Copy1], param->in_bufs[0], param->out_buf);

        for (uint32_t idx = 0; idx < _camera_num; ++idx) {
            start_geomapper (_geomapper[idx][Copy0], param->in_bufs[idx], param->out_buf);

            if (_fastmap_blend_activated) {
                _geomap_buf[idx][BlendLeft] = param->in_bufs[idx];
                _geomap_buf[idx][BlendRight] = param->in_bufs[idx];
            } else {
                _geomap_buf[idx][BlendLeft] = _geomap_pool[idx][BlendLeft]->get_buffer ();
                start_geomapper (_geomapper[idx][BlendLeft], param->in_bufs[idx], _geomap_buf[idx][BlendLeft]);

                _geomap_buf[idx][BlendRight] = _geomap_pool[idx][BlendRight]->get_buffer ();
                start_geomapper (_geomapper[idx][BlendRight], param->in_bufs[idx], _geomap_buf[idx][BlendRight]);
            }
        }
    }

#if HAVE_OPENCV
    if (_stitcher->need_feature_match ()) {
        update_geomapper_factors (_geomapper[0][Copy1], 0);

        for (uint32_t idx = 0; idx < _camera_num; ++idx) {
            update_geomapper_factors (_geomapper[idx][Copy0], idx);
            update_geomapper_factors (_geomapper[idx][BlendLeft], idx);
            update_geomapper_factors (_geomapper[idx][BlendRight], idx);

            start_fm_geomapper (param, idx, FMLeft);
            start_fm_geomapper (param, idx, FMRight);

            _fm_left_factor[idx].reset ();
            _fm_right_factor[idx].reset ();
        }
    }
#endif

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_blenders (const SmartPtr<GLStitcher::StitcherParam> &param)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        uint32_t next_idx = (idx + 1) % _camera_num;
        SmartPtr<GLBlender::BlenderParam> blend_param = new GLBlender::BlenderParam (
            _geomap_buf[idx][BlendRight], _geomap_buf[next_idx][BlendLeft], param->out_buf);

        if (_fastmap_blend_activated) {
            ret = _fastmap_blender[idx]->execute_buffer (blend_param, false);
        } else {
            ret = _blender[idx]->execute_buffer (blend_param, false);
        }

        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher execute blender failed, idx: %d", idx);
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

    const Stitcher::ImageOverlapInfo &overlap = _stitcher->get_overlap (idx);

    float left_offsetx = _matcher[idx]->get_current_left_offset_x ();
    Factor left_factor, right_factor;

    uint32_t left_idx = idx;
    float center_x = (float) _stitcher->get_center (left_idx).slice_center_x;

    float feature_center_x = (float)overlap.left.pos_x + (overlap.left.width / 2.0f);

    float range = feature_center_x - center_x;
    XCAM_ASSERT (range > 1.0f);
    right_factor.x = (range + left_offsetx / 2.0f) / range;
    right_factor.y = 1.0f;
    XCAM_ASSERT (right_factor.x > 0.0f && right_factor.x < 2.0f);

    uint32_t right_idx = (idx + 1) % _camera_num;
    center_x = (float) _stitcher->get_center (right_idx).slice_center_x;
    feature_center_x = (float)overlap.right.pos_x + (overlap.right.width / 2.0f);
    range = center_x - feature_center_x;
    XCAM_ASSERT (range > 1.0f);
    left_factor.x = (range + left_offsetx / 2.0f) / range;
    left_factor.y = 1.0f;
    XCAM_ASSERT (left_factor.x > 0.0f && left_factor.x < 2.0f);

    _fm_right_factor[left_idx] = right_factor;
    _fm_left_factor[right_idx] = left_factor;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_feature_matches ()
{
    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        uint32_t next_idx = (idx + 1) % _camera_num;

        XCamReturn ret = start_feature_match (_geomap_buf[idx][FMRight], _geomap_buf[next_idx][FMLeft], idx);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl-stitcher execute feature match failed, idx: %d", idx);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::start_fm_geomapper (
    const SmartPtr<GLStitcher::StitcherParam> &param, uint32_t idx, GeoMapIdx fm_idx)
{
    if (_geomapper[idx][fm_idx].ptr ()) {
        update_geomapper_factors (_geomapper[idx][fm_idx], idx);

        _geomap_buf[idx][fm_idx] = _geomap_pool[idx][fm_idx]->get_buffer ();
        start_geomapper (_geomapper[idx][fm_idx], param->in_bufs[idx], _geomap_buf[idx][fm_idx]);
    } else {
        GeoMapIdx blend_idx = (fm_idx == FMLeft) ? BlendLeft : BlendRight;

        if (_stitcher->complete_stitch ()) {
            _geomap_buf[idx][fm_idx] = _geomap_buf[idx][blend_idx];
        } else {
            _geomap_buf[idx][fm_idx] = _geomap_pool[idx][blend_idx]->get_buffer ();
            start_geomapper (_geomapper[idx][blend_idx], param->in_bufs[idx], _geomap_buf[idx][fm_idx]);
        }
    }

    return XCAM_RETURN_NO_ERROR;
}
#endif

XCamReturn
StitcherImpl::release_geomapper_rsc (uint32_t cam_id, GeoMapIdx idx)
{
    if (_geomapper[cam_id][idx].ptr ()) {
        _geomapper[cam_id][idx]->terminate ();
        _geomapper[cam_id][idx].release ();
    }

    if (_geomap_pool[cam_id][idx].ptr ()) {
        if (_geomap_buf[cam_id][idx].ptr ()) {
            _geomap_buf[cam_id][idx].release ();
        }
        _geomap_pool[cam_id][idx]->stop ();
        _geomap_pool[cam_id][idx].release ();
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::release_unused_rsc ()
{
    static bool have_been_released = false;
    if (have_been_released)
        return XCAM_RETURN_NO_ERROR;

    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        release_geomapper_rsc (idx, FMLeft);
        release_geomapper_rsc (idx, FMRight);
        if (_matcher[idx].ptr ()) {
            _matcher[idx].release ();
        }

        if (_fastmap_blend_activated) {
            release_geomapper_rsc (idx, BlendLeft);
            release_geomapper_rsc (idx, BlendRight);
            if (_blender[idx].ptr ()) {
                _blender[idx]->terminate ();
                _blender[idx].release ();
            }
        }
    }

    have_been_released = true;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::stop ()
{
    for (uint32_t idx = 0; idx < _camera_num; ++idx) {
        release_geomapper_rsc (idx, Copy0);
        release_geomapper_rsc (idx, Copy1);
        release_geomapper_rsc (idx, BlendLeft);
        release_geomapper_rsc (idx, BlendRight);
        release_geomapper_rsc (idx, FMLeft);
        release_geomapper_rsc (idx, FMRight);

        if (_matcher[idx].ptr ()) {
            _matcher[idx].release ();
        }

        if (_blender[idx].ptr ()) {
            _blender[idx]->terminate ();
            _blender[idx].release ();
        }

        if (_fastmap_blender[idx].ptr ()) {
            _fastmap_blender[idx]->terminate ();
            _fastmap_blender[idx].release ();
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<GLGeoMapHandler>
create_geomapper (GeoMapScaleMode mode)
{
    SmartPtr<GLGeoMapHandler> mapper;
    if (mode == ScaleSingleConst)
        mapper = new GLGeoMapHandler ("stitcher_singleconst_remapper");
    else if (mode == ScaleDualConst) {
        mapper = new GLDualConstGeoMapHandler ("stitcher_dualconst_remapper");
    } else {
        XCAM_LOG_ERROR ("gl-stitcher unsupported geomap scale mode: %d", mode);
    }
    XCAM_ASSERT (mapper.ptr ());

    return mapper;
}

XCamReturn
StitcherImpl::init_geomapper (
    SmartPtr<GLGeoMapHandler> &mapper, const SmartPtr<GLBuffer> &lut, const Stitcher::RoundViewSlice &slice)
{
    XCAM_ASSERT (lut.ptr ());

    mapper = create_geomapper (_stitcher->get_scale_mode ());
    mapper->enable_allocator (false);
    mapper->set_std_output_size (slice.width, slice.height);
    mapper->set_lut_buf (lut);
    mapper->init_factors ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_geomappers (uint32_t idx)
{
    if (_dewarp_mode == DewarpSphere)
        _fisheye_info[idx] = _stitch_info.fisheye_info[idx];

    const Stitcher::RoundViewSlice &slice = _stitcher->get_round_view_slice (idx);
    uint32_t lut_width = XCAM_ALIGN_UP (slice.width / MAP_FACTOR_X, 4);
    uint32_t lut_height = XCAM_ALIGN_UP (slice.height / MAP_FACTOR_Y, 2);

    FisheyeDewarp::MapTable map_table (lut_width * lut_height);
    XCamReturn ret = gen_geomap_table (slice, idx, map_table, lut_width, lut_height);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher generate geomap table failed, idx: %d", idx);

    _geomapper[idx][Copy0] = create_geomapper (_stitcher->get_scale_mode ());
    _geomapper[idx][Copy0]->enable_allocator (false);
    _geomapper[idx][Copy0]->set_std_output_size (slice.width, slice.height);
    _geomapper[idx][Copy0]->set_lookup_table (map_table.data (), lut_width, lut_height);
    _geomapper[idx][Copy0]->init_factors ();

    const SmartPtr<GLBuffer> &lut_buf = _geomapper[idx][Copy0]->get_lut_buf ();
    if (idx == 0) {
        init_geomapper (_geomapper[idx][Copy1], lut_buf, slice);
    }
    init_geomapper (_geomapper[idx][BlendLeft], lut_buf, slice);
    init_geomapper (_geomapper[idx][BlendRight], lut_buf, slice);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
create_buffer_pool (
    SmartPtr<BufferPool> &geomap_pool, const Rect &area)
{
    VideoBufferInfo info;
    info.init (
        V4L2_PIX_FMT_NV12, area.width, area.height,
        XCAM_ALIGN_UP (area.width, GL_STITCHER_ALIGNMENT_X),
        XCAM_ALIGN_UP (area.height, GL_STITCHER_ALIGNMENT_Y));

    SmartPtr<BufferPool> pool = new GLVideoBufferPool (info);
    XCAM_ASSERT (pool.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, pool->reserve (XCAM_GL_RESERVED_BUF_COUNT), XCAM_RETURN_ERROR_MEM,
        "gl-stitcher reserve geomap buffer pool failed, buffer size: %dx%d",
        info.width, info.height);
    geomap_pool = pool;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::config_geomapper_from_blend (
    const Stitcher::ImageOverlapInfo &overlap, uint32_t idx)
{
    _geomapper[idx][BlendRight]->set_std_area (overlap.left);
    _geomapper[idx][BlendRight]->set_extended_offset (0);
    _geomapper[idx][BlendRight]->set_output_size (overlap.out_area.width, overlap.out_area.height);
    XCamReturn ret = create_buffer_pool (_geomap_pool[idx][BlendRight], overlap.out_area);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-stitcher create buffer pool failed, idx: %d", idx);

    uint32_t next_idx = (idx + _camera_num + 1) % _camera_num;
    _geomapper[next_idx][BlendLeft]->set_std_area (overlap.right);
    _geomapper[next_idx][BlendLeft]->set_extended_offset (0);
    _geomapper[next_idx][BlendLeft]->set_output_size (overlap.out_area.width, overlap.out_area.height);
    ret = create_buffer_pool (_geomap_pool[next_idx][BlendLeft], overlap.out_area);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-stitcher create buffer pool failed, idx: %d", next_idx);

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

    config_geomapper_from_blend (overlap, idx);

    overlap.left.pos_x = 0;
    overlap.right.pos_x = 0;

    blender->set_merge_window (overlap.out_area);
    blender->set_input_merge_area (overlap.left, 0);
    blender->set_input_merge_area (overlap.right, 1);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::config_geomappers_from_copy ()
{
    uint32_t width, height;
    _stitcher->get_output_size (width, height);

    Stitcher::CopyAreaArray areas = _stitcher->get_copy_area ();

    uint32_t counter[XCAM_STITCH_MAX_CAMERAS + 1] = {0};
    for (uint32_t idx = 0; idx < areas.size (); ++idx) {
        Stitcher::CopyArea &area = areas[idx];

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

        GeoMapIdx copy_idx = (counter[area.in_idx] == 0) ? Copy0 : Copy1;
        counter[area.in_idx]++;

        SmartPtr<GLGeoMapHandler> &geomapper = _geomapper[area.in_idx][copy_idx];
        geomapper->set_std_area (area.in_area);
        geomapper->set_extended_offset (area.out_area.pos_x);
        geomapper->set_output_size (width, height);
    }

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
StitcherImpl::config_geomapper_from_fm (
    const Stitcher::ImageOverlapInfo &overlap, uint32_t idx, GeoMapIdx fm_idx)
{
    const Stitcher::RoundViewSlice &slice = _stitcher->get_round_view_slice (idx);
    const SmartPtr<GLBuffer> &lut_buf = _geomapper[idx][Copy0]->get_lut_buf ();

    init_geomapper (_geomapper[idx][fm_idx], lut_buf, slice);

    const Rect area = (fm_idx == FMRight) ? overlap.left : overlap.right;
    _geomapper[idx][fm_idx]->set_std_area (area);
    _geomapper[idx][fm_idx]->set_extended_offset (0);
    _geomapper[idx][fm_idx]->set_output_size (overlap.out_area.width, overlap.out_area.height);
    XCamReturn ret = create_buffer_pool (_geomap_pool[idx][fm_idx], overlap.out_area);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "gl-stitcher create buffer pool failed, idx: %d", idx);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
StitcherImpl::init_feature_match (uint32_t idx)
{
    SmartPtr<FeatureMatch> &matcher = _matcher[idx];
    XCamReturn ret = create_feature_match (matcher);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl-stitcher init feature match failed, idx: %d", idx);

    matcher->set_config (_stitcher->get_fm_config ());
    matcher->set_fm_index (idx);

    const BowlDataConfig &bowl = _stitcher->get_bowl_config ();
    const Stitcher::ImageOverlapInfo &overlap = _stitcher->get_overlap (idx);
    const Rect &merge_area = _blender[idx]->get_merge_window ();

    if (!(merge_area.width == overlap.out_area.width)) {
        uint32_t next_idx = (idx + _camera_num + 1) % _camera_num;
        config_geomapper_from_fm (overlap, idx, FMRight);
        config_geomapper_from_fm (overlap, next_idx, FMLeft);
    }

    Rect left = overlap.left;
    Rect right = overlap.right;

    left.pos_x = 0;
    right.pos_x = 0;
    if (_dewarp_mode == DewarpSphere) {
        const FMRegionRatio &ratio = _stitcher->get_fm_region_ratio ();
        left.pos_y = left.height * ratio.pos_y;
        left.height = left.height * ratio.height;
        right.pos_y = left.pos_y;
        right.height = left.height;
    } else {
        left.pos_y = 0;
        left.height =
            int32_t (bowl.wall_height / (bowl.wall_height + bowl.ground_length) * left.height);
        right.pos_y = 0;
        right.height = left.height;
    }

    matcher->set_crop_rect (left, right);

    return XCAM_RETURN_NO_ERROR;
}

#endif

XCamReturn
StitcherImpl::gen_geomap_table (
    const Stitcher::RoundViewSlice &view_slice, uint32_t cam_idx, FisheyeDewarp::MapTable &map_table,
    uint32_t table_width, uint32_t table_height)
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
    dewarper->set_table_size (table_width, table_height);
    dewarper->gen_table (map_table);

    return XCAM_RETURN_NO_ERROR;
}

bool
StitcherImpl::update_geomapper_factors (const SmartPtr<GLGeoMapHandler> &mapper, uint32_t idx)
{
    Factor last_left_factor, last_right_factor, cur_left, cur_right;
    mapper->get_left_factors (last_left_factor.x, last_left_factor.y);
    mapper->get_right_factors (last_right_factor.x, last_right_factor.y);

    cur_left.x = last_left_factor.x * _fm_left_factor[idx].x;
    cur_left.y = last_left_factor.y * _fm_left_factor[idx].y;
    cur_right.x = last_right_factor.x * _fm_right_factor[idx].x;
    cur_right.y = last_right_factor.y * _fm_right_factor[idx].y;

    mapper->update_factors (cur_left.x, cur_left.y, cur_right.x, cur_right.y);

    return true;
}

} // GLStitcherPriv

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
    terminate ();
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

    ensure_stitch_path ();

    SmartPtr<StitcherParam> param = new StitcherParam;
    param->out_buf = out_buf;

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

XCamReturn
GLStitcher::start_work (const SmartPtr<Parameters> &base)
{
    XCAM_ASSERT (base.ptr ());

    SmartPtr<StitcherParam> param = base.dynamic_cast_ptr<StitcherParam> ();
    XCAM_FAIL_RETURN (
        ERROR, param.ptr () && param->in_bufs[0].ptr (), XCAM_RETURN_ERROR_MEM,
        "gl-stitcher execute failed, invalid parameters");

    XCamReturn ret = _impl->start_geomappers (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "gl_stitcher execute geomappers failed");

    if (complete_stitch ()) {
        ret = _impl->start_blenders (param);
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl_stitcher execute blenders failed");
    }

    GLSync::flush ();

#if HAVE_OPENCV
    if (need_feature_match ()) {
        ret = _impl->start_feature_matches ();
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "gl_stitcher execute feature matches failed");
    }
#endif

    if (!need_feature_match ()) {
        _impl->release_unused_rsc ();
    }

    return ret;
}

SmartPtr<Stitcher>
Stitcher::create_gl_stitcher ()
{
    return new GLStitcher;
}

}
