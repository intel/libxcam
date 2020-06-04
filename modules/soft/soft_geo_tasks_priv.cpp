/*
 * soft_geo_tasks_priv.cpp - soft geometry map tasks
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

#include "soft_geo_tasks_priv.h"

namespace XCam {

namespace XCamSoftTasks {

enum BoundState {
    BoundInternal = 0,
    BoundCritical,
    BoundExternal
};

inline void check_bound (const uint32_t &img_w, const uint32_t &img_h, Float2 *in_pos,
                         const uint32_t &max_idx, BoundState &bound)
{
    if (in_pos[0].x >= 0.0f && in_pos[max_idx].x >= 0.0f && in_pos[0].x < img_w && in_pos[max_idx].x < img_w &&
            in_pos[0].y >= 0.0f && in_pos[max_idx].y >= 0.0f && in_pos[0].y < img_h && in_pos[max_idx].y < img_h)
        bound = BoundInternal;
    else if ((in_pos[0].x < 0.0f && in_pos[max_idx].x < 0.0f) || (in_pos[0].x >= img_w && in_pos[max_idx].x >= img_w) ||
             (in_pos[0].y < 0.0f && in_pos[max_idx].y < 0.0f) || (in_pos[0].y >= img_h && in_pos[max_idx].y >= img_h))
        bound = BoundExternal;
    else
        bound = BoundCritical;
}

inline void check_interp_bound (const uint32_t &img_w, const uint32_t &img_h, Float2 *in_pos,
                                const uint32_t &max_idx, BoundState &bound)
{
    for (uint32_t i = 0; i <= max_idx; i++) {
        if (in_pos[i].x < 0.0f) in_pos[i].x = 0.0f;
        if (in_pos[i].x >= img_w - max_idx) in_pos[i].x = img_w - max_idx - 1;
        if (in_pos[i].y < 0.0f) in_pos[i].y = 0.0f;
        if (in_pos[i].y >= img_h) in_pos[i].y = img_h - 1;
    }

    if (in_pos[0].x >= 0.0f && in_pos[max_idx].x >= 0.0f &&
            in_pos[0].x < img_w - max_idx && in_pos[max_idx].x < img_w - max_idx &&
            in_pos[0].y >= 0.0f && in_pos[max_idx].y >= 0.0f && in_pos[0].y < img_h && in_pos[max_idx].y < img_h)
        bound = BoundInternal;
    else if ((in_pos[0].x < 0.0f && in_pos[max_idx].x < 0.0f) || (in_pos[0].x >= img_w && in_pos[max_idx].x >= img_w) ||
             (in_pos[0].y < 0.0f && in_pos[max_idx].y < 0.0f) || (in_pos[0].y >= img_h && in_pos[max_idx].y >= img_h))
        bound = BoundExternal;
    else
        bound = BoundCritical;
}

template <typename TypeT>
inline void calc_critical_pixels (const uint32_t &img_w, const uint32_t &img_h, Float2 *in_pos,
                                  const uint32_t &max_idx, const TypeT &zero_byte, TypeT *luma)
{
    for (uint32_t idx = 0; idx < max_idx; ++idx) {
        if (in_pos[idx].x < 0.0f || in_pos[idx].x >= img_w || in_pos[idx].y < 0.0f || in_pos[idx].y >= img_h)
            luma[idx] = zero_byte;
    }
}

static void interp_sample_pos (const Float2Image *lut, Float2* interp_pos, const Float2 &first, const Float2 &step)
{
#if ENABLE_AVX512
    Float2 lut_pos[16];
    __m512 x512 = _mm512_set1_ps(step.x);
    __m512 multiplier = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f);
    x512 = _mm512_fmadd_ps(x512, multiplier, _mm512_set1_ps(first.x));
    __m512 y512 = _mm512_set1_ps(first.y);
    __m512 Lo = _mm512_unpacklo_ps(x512, y512);
    __m512 Hi = _mm512_unpackhi_ps(x512, y512);

    __m512i idx0 = _mm512_setr_epi32(0, 1, 2, 3, 0x10, 0x11, 0x12, 0x13, 4, 5, 6, 7, 0x14, 0x15, 0x16, 0x17);
    __m512i idx1 = _mm512_setr_epi32(8, 9, 0xa, 0xb, 0x18, 0x19, 0x1a, 0x1b, 0xc, 0xd, 0xe, 0xf, 0x1c, 0x1d, 0x1e, 0x1f);
    __m512 data0 = _mm512_permutex2var_ps(Lo, idx0, Hi);
    __m512 data1 = _mm512_permutex2var_ps(Lo, idx1, Hi);
    _mm512_storeu_ps(lut_pos, data0);
    _mm512_storeu_ps(&lut_pos[8], data1);
#else
    Float2 lut_pos[16] = {
        first, Float2(first.x + step.x, first.y),
        Float2(first.x + step.x * 2, first.y), Float2(first.x + step.x * 3, first.y),
        Float2(first.x + step.x * 4, first.y), Float2(first.x + step.x * 5, first.y),
        Float2(first.x + step.x * 6, first.y), Float2(first.x + step.x * 7, first.y),
        Float2(first.x + step.x * 8, first.y), Float2(first.x + step.x * 9, first.y),
        Float2(first.x + step.x * 10, first.y), Float2(first.x + step.x * 11, first.y),
        Float2(first.x + step.x * 12, first.y), Float2(first.x + step.x * 13, first.y),
        Float2(first.x + step.x * 14, first.y), Float2(first.x + step.x * 15, first.y)
    };
#endif
#if ENABLE_AVX512
    BoundState interp_bound = BoundInternal;
    check_interp_bound (lut->get_width (), lut->get_height (), interp_pos, XCAM_SOFT_WORKUNIT_PIXELS - 1, interp_bound);
    if (interp_bound == BoundInternal) {
        lut->read_interpolate_array (lut_pos, interp_pos);
    } else {
        lut->read_interpolate_array<Float2, XCAM_SOFT_WORKUNIT_PIXELS> (lut_pos, interp_pos);
    }
#else
    lut->read_interpolate_array<Float2, XCAM_SOFT_WORKUNIT_PIXELS> (lut_pos, interp_pos);
#endif
}

static void map_image (
    const UcharImage *in, UcharImage *out, Float2 *interp_pos,
    const uint32_t &width, const uint32_t &height,
    const uint32_t &out_x, const uint32_t &out_y,
    const Uchar *zero_byte, const bool is_chroma = false)
{
    float  interp_value[XCAM_SOFT_WORKUNIT_PIXELS];
    Uchar  interp_pixel_vaule[XCAM_SOFT_WORKUNIT_PIXELS];
    BoundState bound = BoundInternal;

    if (is_chroma) {
        check_bound (width, height, interp_pos, XCAM_SOFT_WORKUNIT_PIXELS / 2 - 1, bound);
    } else {
        check_bound (width, height, interp_pos, XCAM_SOFT_WORKUNIT_PIXELS - 1, bound);
    }

    if (bound == BoundExternal) {
        if (is_chroma) {
            out->write_array_no_check < XCAM_SOFT_WORKUNIT_PIXELS / 2 > (out_x, out_y, zero_byte);
        } else {
            out->write_array_no_check<XCAM_SOFT_WORKUNIT_PIXELS> (out_x, out_y, zero_byte);
        }
    } else {
#if ENABLE_AVX512
        BoundState interp_bound = BoundInternal;
        if (is_chroma) {
            check_interp_bound (in->get_width (), in->get_height (), interp_pos, XCAM_SOFT_WORKUNIT_PIXELS / 2 - 1, interp_bound);
        } else {
            check_interp_bound (in->get_width (), in->get_height (), interp_pos, XCAM_SOFT_WORKUNIT_PIXELS - 1, interp_bound);
        }
        if (interp_bound == BoundInternal) {
            in->read_interpolate_array (interp_pos, interp_pixel_vaule, is_chroma);
        } else {
            if (is_chroma) {
                in->read_interpolate_array < float, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_pos, interp_value);
                convert_to_uchar_N < float, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_value, interp_pixel_vaule);
            } else {
                in->read_interpolate_array<float, XCAM_SOFT_WORKUNIT_PIXELS> (interp_pos, interp_value);
                convert_to_uchar_N<float, XCAM_SOFT_WORKUNIT_PIXELS> (interp_value, interp_pixel_vaule);
            }
        }
#else
        if (is_chroma) {
            in->read_interpolate_array < float, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_pos, interp_value);
            convert_to_uchar_N < float, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_value, interp_pixel_vaule);
        } else {
            in->read_interpolate_array<float, XCAM_SOFT_WORKUNIT_PIXELS> (interp_pos, interp_value);
            convert_to_uchar_N<float, XCAM_SOFT_WORKUNIT_PIXELS> (interp_value, interp_pixel_vaule);
        }
#endif
        if (bound == BoundCritical) {
            if (is_chroma) {
                calc_critical_pixels (width, height, interp_pos, XCAM_SOFT_WORKUNIT_PIXELS / 2, zero_byte[0], interp_pixel_vaule);
            } else {
                calc_critical_pixels (width, height, interp_pos, XCAM_SOFT_WORKUNIT_PIXELS, zero_byte[0], interp_pixel_vaule);
            }
        }
        if (is_chroma) {
            out->write_array_no_check < XCAM_SOFT_WORKUNIT_PIXELS / 2 > (out_x, out_y, interp_pixel_vaule);
        } else {
            out->write_array_no_check<XCAM_SOFT_WORKUNIT_PIXELS> (out_x, out_y, interp_pixel_vaule);
        }
    }
}

static void map_image (
    const Uchar2Image *in, Uchar2Image *out, Float2 *interp_pos,
    const uint32_t &width, const uint32_t &height,
    const uint32_t &out_x, const uint32_t &out_y,
    const Uchar2 *zero_byte)
{
    BoundState bound = BoundInternal;

    Float2 interp_value[XCAM_SOFT_WORKUNIT_PIXELS / 2];
    Uchar2 interp_pixel_value[XCAM_SOFT_WORKUNIT_PIXELS / 2];

#if ENABLE_AVX512
    XCAM_ASSERT (XCAM_SOFT_WORKUNIT_PIXELS == 16);
    __m512i index = _mm512_setr_epi32 (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29);
    __m512 multiplier = _mm512_set1_ps (0.5f);
    __m512 value = _mm512_i32gather_ps (index, interp_pos, 4);
    value = _mm512_mul_ps (value, multiplier);
    _mm512_storeu_ps (interp_pos, value);
#else
    for (uint32_t i = 0; i < XCAM_SOFT_WORKUNIT_PIXELS; i += 2) {
        interp_pos[i / 2] = interp_pos[i] / 2.0f;
    }
#endif

    check_bound (width, height, interp_pos, XCAM_SOFT_WORKUNIT_PIXELS / 2 - 1, bound);
    if (bound == BoundExternal) {
        out->write_array_no_check < XCAM_SOFT_WORKUNIT_PIXELS / 2 > (out_x, out_y, zero_byte);
    }
    else {
#if ENABLE_AVX512
        BoundState interp_bound = BoundInternal;
        check_interp_bound (in->get_width (), in->get_height (), interp_pos, XCAM_SOFT_WORKUNIT_PIXELS / 2 - 1, interp_bound);
        if (interp_bound == BoundInternal) {
            in->read_interpolate_array (interp_pos, interp_pixel_value);
        } else {
            in->read_interpolate_array < Float2, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_pos, interp_value);
            convert_to_uchar2_N < Float2, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_value, interp_pixel_value);
        }
#else
        in->read_interpolate_array < Float2, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_pos, interp_value);
        convert_to_uchar2_N < Float2, XCAM_SOFT_WORKUNIT_PIXELS / 2 > (interp_value, interp_pixel_value);
#endif
        if (bound == BoundCritical) {
            calc_critical_pixels (width, height, interp_pos, XCAM_SOFT_WORKUNIT_PIXELS / 2, zero_byte[0], interp_pixel_value);
        }
        out->write_array_no_check < XCAM_SOFT_WORKUNIT_PIXELS / 2 > (out_x, out_y, interp_pixel_value);
    }

}

XCamReturn
GeoMapTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    static const Uchar zero_luma_byte[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static const Uchar2 zero_uv_byte[8] = {{128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}};
    static const Uchar zero_chroma_byte[8] = {128, 128, 128, 128, 128, 128, 128, 128};

    SmartPtr<GeoMapTask::Args> args = base.dynamic_cast_ptr<GeoMapTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    UcharImage *in_luma = args->in_luma.ptr ();
    UcharImage *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = NULL;
    Uchar2Image *out_uv = NULL;
    UcharImage *in_u = NULL;
    UcharImage *in_v = NULL;
    UcharImage *out_u = NULL;
    UcharImage *out_v = NULL;
    if (NULL != args->in_uv.ptr ()) {
        in_uv = args->in_uv.ptr ();
        out_uv = args->out_uv.ptr ();
    } else if (NULL != args->in_u.ptr () && NULL != args->in_v.ptr ()) {
        in_u = args->in_u.ptr ();
        out_u = args->out_u.ptr ();
        in_v = args->in_v.ptr ();
        out_v = args->out_v.ptr ();
    }

    Float2Image *lut = args->lookup_table.ptr ();
    XCAM_ASSERT (in_luma && (in_uv || (in_u && in_v)));
    XCAM_ASSERT (out_luma && (out_uv || (out_u && out_v)));
    XCAM_ASSERT (lut);

    Float2 factors = args->factors;
    XCAM_ASSERT (!XCAM_DOUBLE_EQUAL_AROUND (factors.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factors.y, 0.0f));

    Float2 step = Float2(1.0f, 1.0f) / factors;

    Float2 out_center ((out_luma->get_width () - 1.0f ) / 2.0f, (out_luma->get_height () - 1.0f ) / 2.0f);
    Float2 lut_center ((lut->get_width () - 1.0f) / 2.0f, (lut->get_height () - 1.0f) / 2.0f);

    uint32_t luma_w = in_luma->get_width ();
    uint32_t luma_h = in_luma->get_height ();
    uint32_t chroma_w = luma_w / 2;
    uint32_t chroma_h = luma_h / 2;
    if (NULL != in_uv) {
        chroma_w = in_uv->get_width ();
        chroma_h = in_uv->get_height ();
    } else if (NULL != in_u && NULL != in_v) {
        chroma_w = in_u->get_width ();
        chroma_h = in_u->get_height ();
    }

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y) {
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x) {
            uint32_t out_x = x * XCAM_SOFT_WORKUNIT_PIXELS, out_y = y * 2;

            // calculate XCAM_SOFT_WORKUNIT_PIXELS * 2 luma, center aligned
            Float2 out_pos (out_x, out_y);
            out_pos -= out_center;
            Float2 first = out_pos / factors;
            first += lut_center;

            Float2 interp_pos[XCAM_SOFT_WORKUNIT_PIXELS] = { Float2(0.0f, 0.0f) };

            if (NULL != in_u && NULL != in_v) {
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y, zero_luma_byte);

                for (uint32_t i = 0; i < XCAM_SOFT_WORKUNIT_PIXELS; i += 2) {
                    interp_pos[i / 2] = interp_pos[i] / 2.0f;
                }
                map_image (in_u, out_u, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_chroma_byte, true);

                map_image (in_v, out_v, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_chroma_byte, true);

                first.y = first.y + step.y;
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y + 1, zero_luma_byte);
            } else if (NULL != in_uv) {
                interp_sample_pos (lut, interp_pos, first, step);

                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y, zero_luma_byte);

                map_image (in_uv, out_uv, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_uv_byte);

                first.y = first.y + step.y;
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y + 1, zero_luma_byte);
            }
        }
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GeoMapDualConstTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    static const Uchar zero_luma_byte[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static const Uchar2 zero_uv_byte[8] = {{128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}};
    static const Uchar zero_chroma_byte[8] = {128, 128, 128, 128, 128, 128, 128, 128};

    SmartPtr<GeoMapDualConstTask::Args> args = base.dynamic_cast_ptr<GeoMapDualConstTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    UcharImage *in_luma = args->in_luma.ptr ();
    UcharImage *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = NULL;
    Uchar2Image *out_uv = NULL;
    UcharImage *in_u = NULL;
    UcharImage *in_v = NULL;
    UcharImage *out_u = NULL;
    UcharImage *out_v = NULL;
    if (NULL != args->in_uv.ptr ()) {
        in_uv = args->in_uv.ptr ();
        out_uv = args->out_uv.ptr ();
    } else if (NULL != args->in_u.ptr () && NULL != args->in_v.ptr ()) {
        in_u = args->in_u.ptr ();
        out_u = args->out_u.ptr ();
        in_v = args->in_v.ptr ();
        out_v = args->out_v.ptr ();
    }
    Float2Image *lut = args->lookup_table.ptr ();
    XCAM_ASSERT (in_luma && (in_uv || (in_u && in_v)));
    XCAM_ASSERT (out_luma && (out_uv || (out_u && out_v)));
    XCAM_ASSERT (lut);

    Float2 left_factor = args->left_factor;
    Float2 right_factor = args->right_factor;
    XCAM_ASSERT (
        !XCAM_DOUBLE_EQUAL_AROUND (left_factor.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (left_factor.y, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (right_factor.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (right_factor.y, 0.0f));

    Float2 left_step = Float2(1.0f, 1.0f) / left_factor;
    Float2 right_step = Float2(1.0f, 1.0f) / right_factor;

    Float2 out_center ((out_luma->get_width () - 1.0f ) / 2.0f, (out_luma->get_height () - 1.0f ) / 2.0f);
    Float2 lut_center ((lut->get_width () - 1.0f) / 2.0f, (lut->get_height () - 1.0f) / 2.0f);

    uint32_t luma_w = in_luma->get_width ();
    uint32_t luma_h = in_luma->get_height ();
    uint32_t chroma_w = luma_w / 2;
    uint32_t chroma_h = luma_h / 2;
    if (NULL != in_uv) {
        chroma_w = in_uv->get_width ();
        chroma_h = in_uv->get_height ();
    } else if (NULL != in_u && NULL != in_v) {
        chroma_w = in_u->get_width ();
        chroma_h = in_u->get_height ();
    }

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y) {
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x) {
            uint32_t out_x = x * XCAM_SOFT_WORKUNIT_PIXELS, out_y = y * 2;
            Float2 &factor = (out_x + XCAM_SOFT_WORKUNIT_PIXELS / 2 < out_center.x) ? left_factor : right_factor;
            Float2 &step = (out_x + XCAM_SOFT_WORKUNIT_PIXELS / 2 < out_center.x) ? left_step : right_step;

            // calculate XCAM_SOFT_WORKUNIT_PIXELS * 2 luma, center aligned
            Float2 out_pos (out_x, out_y);
            out_pos -= out_center;
            Float2 first = out_pos / factor;
            first += lut_center;

            Float2 interp_pos[XCAM_SOFT_WORKUNIT_PIXELS] = { Float2(0.0f, 0.0f) };
            if (NULL != in_u && NULL != in_v) {
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y, zero_luma_byte);

                for (uint32_t i = 0; i < XCAM_SOFT_WORKUNIT_PIXELS; i += 2) {
                    interp_pos[i / 2] = interp_pos[i] / 2.0f;
                }
                map_image (in_u, out_u, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_chroma_byte, true);

                map_image (in_v, out_v, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_chroma_byte, true);

                first.y = first.y + step.y;
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y + 1, zero_luma_byte);
            } else if (NULL != in_uv) {
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y, zero_luma_byte);

                map_image (in_uv, out_uv, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_uv_byte);

                first.y = first.y + step.y;
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y + 1, zero_luma_byte);
            }
        }
    }
    return XCAM_RETURN_NO_ERROR;
}

GeoMapDualCurveTask::GeoMapDualCurveTask (const SmartPtr<Worker::Callback> &cb)
    : GeoMapDualConstTask (cb)
    , _scaled_height (0.0f)
    , _left_std_factor (0.0f, 0.0f)
    , _right_std_factor (0.0f, 0.0f)
    , _left_factors (NULL)
    , _right_factors (NULL)
    , _left_steps (NULL)
    , _right_steps (NULL)
    , _initialized (false)
{
    set_work_unit (XCAM_SOFT_WORKUNIT_PIXELS, 2);
}

GeoMapDualCurveTask::~GeoMapDualCurveTask () {
    if (_left_factors) {
        delete [] _left_factors;
        _left_factors = NULL;
    }
    if (_right_factors) {
        delete [] _right_factors;
        _right_factors = NULL;
    }
    if (_left_steps) {
        delete [] _left_steps;
        _left_steps = NULL;
    }
    if (_right_steps) {
        delete [] _right_steps;
        _right_steps = NULL;
    }
}

void
GeoMapDualCurveTask::set_left_std_factor (float x, float y) {
    XCAM_ASSERT (!XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f));

    _left_std_factor.x = x;
    _left_std_factor.y = y;
}

void
GeoMapDualCurveTask::set_right_std_factor (float x, float y) {
    XCAM_ASSERT (!XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f));

    _right_std_factor.x = x;
    _right_std_factor.y = y;
}

static void calc_cur_row_factor (
    const uint32_t &y, const uint32_t &ym,
    const Float2 &std_factor, const float &scaled_height,
    const Float2 &factor, Float2 &cur_row_factor)
{
    float a, b, c;
    a = (std_factor.x - factor.x) / ((scaled_height - ym) * (scaled_height - ym));
    b = -2 * a * ym;
    c = std_factor.x - a * scaled_height * scaled_height - b * scaled_height;
    cur_row_factor.x = (y >= scaled_height) ? std_factor.x : ((y < ym) ? factor.x : (a * y * y + b * y + c));

    cur_row_factor.y = factor.y;
}

bool
GeoMapDualCurveTask::set_factors (SmartPtr<GeoMapDualCurveTask::Args> args, uint32_t size) {
    if (!_initialized) {
        SmartLock locker (_mutex);

        if (_left_factors == NULL) {
            _left_factors = new Float2[size];
            XCAM_ASSERT (_left_factors);
        }
        if (_right_factors == NULL) {
            _right_factors = new Float2[size];
            XCAM_ASSERT (_right_factors);
        }
        if (_left_steps == NULL) {
            _left_steps =  new Float2[size];
            XCAM_ASSERT (_left_steps);
        }
        if (_right_steps == NULL) {
            _right_steps =  new Float2[size];
            XCAM_ASSERT (_right_steps);
        }
        _initialized = true;
    }

    float ym = _scaled_height * 0.5f;
    for (uint32_t y = 0; y < size; ++y) {
        calc_cur_row_factor (y, ym, _left_std_factor, _scaled_height, args->left_factor, _left_factors[y]);
        calc_cur_row_factor (y, ym, _right_std_factor, _scaled_height, args->right_factor, _right_factors[y]);
    }

    for (uint32_t y = 0; y < size; ++y) {
        XCAM_FAIL_RETURN (
            ERROR,
            !XCAM_DOUBLE_EQUAL_AROUND (_left_factors[y].x, 0.0f) &&
            !XCAM_DOUBLE_EQUAL_AROUND (_left_factors[y].y, 0.0f) &&
            !XCAM_DOUBLE_EQUAL_AROUND (_right_factors[y].x, 0.0f) &&
            !XCAM_DOUBLE_EQUAL_AROUND (_right_factors[y].y, 0.0f),
            false,
            "GeoMapDualCurveTask invalid factor(row:%d): left_factor(x:%f, y:%f) right_factor(x:%f, y:%f)",
            y, _left_factors[y].x, _left_factors[y].y, _right_factors[y].x, _right_factors[y].y);

        _left_steps[y] = Float2(1.0f, 1.0f) / _left_factors[y];
        _right_steps[y] = Float2(1.0f, 1.0f) / _right_factors[y];
    }

    return true;
}

XCamReturn
GeoMapDualCurveTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    static const Uchar zero_luma_byte[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static const Uchar2 zero_uv_byte[8] = {{128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}};
    static const Uchar zero_chroma_byte[8] = {128, 128, 128, 128, 128, 128, 128, 128};

    SmartPtr<GeoMapDualCurveTask::Args> args = base.dynamic_cast_ptr<GeoMapDualCurveTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    XCAM_ASSERT (
        !XCAM_DOUBLE_EQUAL_AROUND (args->left_factor.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (args->left_factor.y, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (args->right_factor.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (args->right_factor.y, 0.0f));

    UcharImage *in_luma = args->in_luma.ptr ();
    UcharImage *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = NULL;
    Uchar2Image *out_uv = NULL;
    UcharImage *in_u = NULL;
    UcharImage *in_v = NULL;
    UcharImage *out_u = NULL;
    UcharImage *out_v = NULL;
    if (NULL != args->in_uv.ptr ()) {
        in_uv = args->in_uv.ptr ();
        out_uv = args->out_uv.ptr ();
    } else if (NULL != args->in_u.ptr () && NULL != args->in_v.ptr ()) {
        in_u = args->in_u.ptr ();
        out_u = args->out_u.ptr ();
        in_v = args->in_v.ptr ();
        out_v = args->out_v.ptr ();
    }

    Float2Image *lut = args->lookup_table.ptr ();
    XCAM_ASSERT (in_luma && (in_uv || (in_u && in_v)));
    XCAM_ASSERT (out_luma && (out_uv || (out_u && out_v)));
    XCAM_ASSERT (lut);

    set_factors (args, out_luma->get_height ());

    Float2 out_center ((out_luma->get_width () - 1.0f ) / 2.0f, (out_luma->get_height () - 1.0f ) / 2.0f);
    Float2 lut_center ((lut->get_width () - 1.0f) / 2.0f, (lut->get_height () - 1.0f) / 2.0f);

    uint32_t luma_w = in_luma->get_width ();
    uint32_t luma_h = in_luma->get_height ();
    uint32_t chroma_w = luma_w / 2;
    uint32_t chroma_h = luma_h / 2;
    if (NULL != in_uv) {
        chroma_w = in_uv->get_width ();
        chroma_h = in_uv->get_height ();
    } else if (NULL != in_u && NULL != in_v) {
        chroma_w = in_u->get_width ();
        chroma_h = in_u->get_height ();
    }

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y) {
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x) {
            uint32_t out_x = x * XCAM_SOFT_WORKUNIT_PIXELS, out_y = y * 2;
            Float2 &factor = (out_x + XCAM_SOFT_WORKUNIT_PIXELS / 2 < out_center.x) ? _left_factors[out_y] : _right_factors[out_y];
            Float2 &step = (out_x + XCAM_SOFT_WORKUNIT_PIXELS / 2 < out_center.x) ? _left_steps[out_y] : _right_steps[out_y];

            // calculate XCAM_SOFT_WORKUNIT_PIXELS * 2 luma, center aligned
            Float2 out_pos (out_x, out_y);
            out_pos -= out_center;
            Float2 first = out_pos / factor;
            first += lut_center;

            Float2 interp_pos[XCAM_SOFT_WORKUNIT_PIXELS] = { Float2(0.0f, 0.0f) };
            if (NULL != in_u && NULL != in_v) {
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y, zero_luma_byte);

                for (uint32_t i = 0; i < XCAM_SOFT_WORKUNIT_PIXELS; i += 2) {
                    interp_pos[i / 2] = interp_pos[i] / 2.0f;
                }
                map_image (in_u, out_u, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_chroma_byte, true);

                map_image (in_v, out_v, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_chroma_byte, true);

                first.y = first.y + step.y;
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y + 1, zero_luma_byte);
            } else if (NULL != in_uv) {
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y, zero_luma_byte);

                map_image (in_uv, out_uv, interp_pos, chroma_w, chroma_h,
                           out_x / 2, out_y / 2, zero_uv_byte);

                first.y = first.y + step.y;
                interp_sample_pos (lut, interp_pos, first, step);
                map_image (in_luma, out_luma, interp_pos, luma_w, luma_h,
                           out_x, out_y + 1, zero_luma_byte);
            }
        }
    }
    return XCAM_RETURN_NO_ERROR;
}

}

}
