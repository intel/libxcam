/*
 * soft_image.h - soft image class
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

#ifndef XCAM_SOFT_IMAGE_H
#define XCAM_SOFT_IMAGE_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <vec_mat.h>
#include <file_handle.h>

#if ENABLE_AVX512
#include <immintrin.h>
#endif

#if ENABLE_AVX512
#define XCAM_SOFT_WORKUNIT_PIXELS 16
#else
#define XCAM_SOFT_WORKUNIT_PIXELS 8
#endif

namespace XCam {

typedef uint8_t Uchar;
typedef int8_t Char;
typedef Vector2<uint8_t> Uchar2;
typedef Vector2<int8_t> Char2;
typedef Vector2<float> Float2;
typedef Vector2<int> Int2;

enum BorderType {
    BorderTypeNearest,
    BorderTypeConst,
    BorderTypeRewind,
};

template <typename T>
class SoftImage
{
public:
    typedef T Type;
private:
    uint8_t    *_buf_ptr;
    uint32_t    _width;
    uint32_t    _height;
    uint32_t    _pitch;

    SmartPtr<VideoBuffer> _bind;

public:
    explicit SoftImage (const SmartPtr<VideoBuffer> &buf, const uint32_t plane);
    explicit SoftImage (
        const uint32_t width, const uint32_t height,
        uint32_t aligned_width = 0);
    explicit SoftImage (
        const SmartPtr<VideoBuffer> &buf,
        const uint32_t width, const uint32_t height, const uint32_t pictch, const uint32_t offset = 0);

    ~SoftImage () {
        if (!_bind.ptr ()) {
            xcam_free (_buf_ptr);
        }
    }

    uint32_t pixel_size () const {
        return sizeof (T);
    }

    uint32_t get_width () const {
        return _width;
    }
    uint32_t get_height () const {
        return _height;
    }
    uint32_t get_pitch () const {
        return _pitch;
    }
    bool is_valid () const {
        return (_buf_ptr && _width && _height);
    }

    const SmartPtr<VideoBuffer> &get_bind_buf () const {
        return _bind;
    }
    T *get_buf_ptr (int32_t x, int32_t y) {
        return (T *)(_buf_ptr + y * _pitch) + x;
    }
    const T *get_buf_ptr (int32_t x, int32_t y) const {
        return (const T *)(_buf_ptr + y * _pitch) + x;
    }

    inline T read_data_no_check (int32_t x, int32_t y) const {
        const T *t_ptr = (const T *)(_buf_ptr + y * _pitch);
        return t_ptr[x];
    }

    inline T read_data (int32_t x, int32_t y) const {
        border_check (x, y);
        return read_data_no_check (x, y);
    }

    template<typename O>
    inline O read_interpolate_data (float x, float y) const;

    template<typename O, uint32_t N>
    inline void read_interpolate_array (Float2 *pos, O *array) const;

#if ENABLE_AVX512
    inline void read_interpolate_array (Float2 *pos, Float2 *array) const;
    inline void read_interpolate_array (Float2 *pos, Uchar *array, bool is_chroma = false) const;
    inline void read_interpolate_array (Float2 *pos, Uchar2 *array) const;
#endif

    template<uint32_t N>
    inline void read_array_no_check (const int32_t x, const int32_t y, T *array) const {
        XCAM_ASSERT (N <= 8);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch)) + x;
        memcpy (array, t_ptr, sizeof (T) * N);
    }

    template<typename O, uint32_t N>
    inline void read_array_no_check (const int32_t x, const int32_t y, O *array) const {
        XCAM_ASSERT (N <= 8);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch)) + x;
        for (uint32_t i = 0; i < N; ++i) {
            array[i] = t_ptr[i];
        }
    }

    template<uint32_t N>
    inline void read_array (int32_t x, int32_t y, T *array) const {
        XCAM_ASSERT (N <= 8);
        border_check_y (y);
        if (x + N < _width) {
            read_array_no_check<N> (x, y, array);
        } else {
            const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch));
            for (uint32_t i = 0; i < N; ++i, ++x) {
                border_check_x (x);
                array[i] = t_ptr[x];
            }
        }
    }

    template<typename O, uint32_t N>
    inline void read_array (int32_t x, int32_t y, O *array) const {
        XCAM_ASSERT (N <= 8);
        border_check_y (y);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch));
        for (uint32_t i = 0; i < N; ++i, ++x) {
            border_check_x (x);
            array[i] = t_ptr[x];
        }
    }

    inline void write_data (int32_t x, int32_t y, const T &v) {
        if (x < 0 || x >= (int32_t)_width)
            return;
        if (y < 0 || y >= (int32_t)_height)
            return;
        write_data_no_check (x, y, v);
    }

    inline void write_data_no_check (int32_t x, int32_t y, const T &v) {
        T *t_ptr = (T *)(_buf_ptr + y * _pitch);
        t_ptr[x] = v;
    }

    template<uint32_t N>
    inline void write_array_no_check (int32_t x, int32_t y, const T *array) {
        T *t_ptr = (T *)(_buf_ptr + y * _pitch);
        memcpy (t_ptr + x, array, sizeof (T) * N);
    }

    template<uint32_t N>
    inline void write_array (int32_t x, int32_t y, const T *array) {
        if (y < 0 || y >= (int32_t)_height)
            return;

        if (x >= 0 && x + N <= _width) {
            write_array_no_check<N> (x, y, array);
        } else {
            T *t_ptr = ((T *)(_buf_ptr + y * _pitch));
            for (uint32_t i = 0; i < N; ++i, ++x) {
                if (x < 0 || x >= (int32_t)_width) continue;
                t_ptr[x] = array[i];
            }
        }
    }

private:
    inline void border_check_x (int32_t &x) const {
        x = x < 0 ? 0 : x;
        x = x > (int32_t)(_width - 1) ? (int32_t)(_width - 1) : x;
    }

    inline void border_check_y (int32_t &y) const {
        y = y < 0 ? 0 : y;
        y = y > (int32_t)(_height - 1) ? (int32_t)(_height - 1) : y;
    }

    inline void border_check (int32_t &x, int32_t &y) const {
        border_check_x (x);
        border_check_y (y);
    }
};


template <typename T>
SoftImage<T>::SoftImage (const SmartPtr<VideoBuffer> &buf, const uint32_t plane)
    : _buf_ptr (NULL)
    , _width (0), _height (0), _pitch (0)
{
    XCAM_ASSERT (buf.ptr ());
    const VideoBufferInfo &info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    if (!info.get_planar_info(planar, plane)) {
        XCAM_LOG_ERROR (
            "videobuf to soft image failed. buf format:%s, plane:%d", xcam_fourcc_to_string (info.format), plane);
        return;
    }
    _buf_ptr = buf->map () + info.offsets[plane];
    XCAM_ASSERT (_buf_ptr);
    _pitch = info.strides[plane];
    _height = planar.height;
    _width = planar.pixel_bytes * planar.width / sizeof (T);
    XCAM_ASSERT (_width * sizeof(T) == planar.pixel_bytes * planar.width);
    _bind = buf;
}

template <typename T>
SoftImage<T>::SoftImage (
    const uint32_t width, const uint32_t height, uint32_t aligned_width)
    : _buf_ptr (NULL)
    , _width (0), _height (0), _pitch (0)
{
    if (!aligned_width)
        aligned_width = width;

    XCAM_ASSERT (aligned_width >= width);
    XCAM_ASSERT (width > 0 && height > 0);
    _pitch = aligned_width * sizeof (T);
    _buf_ptr = (uint8_t *)xcam_malloc (_pitch * height);
    XCAM_ASSERT (_buf_ptr);
    _width = width;
    _height = height;
}

template <typename T>
SoftImage<T>::SoftImage (
    const SmartPtr<VideoBuffer> &buf,
    const uint32_t width, const uint32_t height, const uint32_t pictch, const uint32_t offset)
    : _buf_ptr (NULL)
    , _width (width), _height (height)
    , _pitch (pictch)
    , _bind (buf)
{
    XCAM_ASSERT (buf.ptr ());
    XCAM_ASSERT (buf->map ());
    _buf_ptr = buf->map () + offset;
}

template <typename T>
inline Uchar convert_to_uchar (const T& v) {
    if (v < 0.0f) return 0;
    else if (v > 255.0f) return 255;
    return (Uchar)(v + 0.5f);
}

template <typename T, uint32_t N>
inline void convert_to_uchar_N (const T *in, Uchar *out) {
    for (uint32_t i = 0; i < N; ++i) {
        out[i] = convert_to_uchar<T> (in[i]);
    }
}

template <typename Vec2>
inline Uchar2 convert_to_uchar2 (const Vec2& v) {
    return Uchar2 (convert_to_uchar(v.x), convert_to_uchar(v.y));
}

template <typename Vec2, uint32_t N>
inline void convert_to_uchar2_N (const Vec2 *in, Uchar2 *out) {
    for (uint32_t i = 0; i < N; ++i) {
        out[i].x = convert_to_uchar (in[i].x);
        out[i].y = convert_to_uchar (in[i].y);
    }
}

typedef SoftImage<Uchar> UcharImage;
typedef SoftImage<Uchar2> Uchar2Image;
typedef SoftImage<float> FloatImage;
typedef SoftImage<Float2> Float2Image;

template <class SoftImageT>
class SoftImageFile
    : public FileHandle
{
public:
    SoftImageFile () {}
    explicit SoftImageFile (const char *name, const char *option)
        : FileHandle (name, option)
    {}

    inline XCamReturn read_buf (const SmartPtr<SoftImageT> &buf);
    inline XCamReturn write_buf (const SmartPtr<SoftImageT> &buf);
};

template <class SoftImageT>
inline XCamReturn
SoftImageFile<SoftImageT>::read_buf (const SmartPtr<SoftImageT> &buf)
{
    XCAM_FAIL_RETURN (
        WARNING, is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) read buf failed, file is not open", XCAM_STR (get_file_name ()));

    XCAM_FAIL_RETURN (
        WARNING, buf->is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) read buf failed, buf is not valid", XCAM_STR (get_file_name ()));

    XCAM_ASSERT (is_valid ());
    uint32_t height = buf->get_height ();
    uint32_t line_bytes = buf->get_width () * buf->pixel_size ();

    for (uint32_t index = 0; index < height; index++) {
        uint8_t *line_ptr = buf->get_buf_ptr (0, index);
        XCAM_FAIL_RETURN (
            WARNING, fread (line_ptr, 1, line_bytes, _fp) == line_bytes, XCAM_RETURN_ERROR_FILE,
            "soft image file(%s) read buf failed, image_line:%d", XCAM_STR (get_file_name ()), index);
    }
    return XCAM_RETURN_NO_ERROR;
}

template <class SoftImageT>
inline XCamReturn
SoftImageFile<SoftImageT>::write_buf (const SmartPtr<SoftImageT> &buf)
{
    XCAM_FAIL_RETURN (
        WARNING, is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) write buf failed, file is not open", XCAM_STR (get_file_name ()));

    XCAM_FAIL_RETURN (
        WARNING, buf->is_valid (), XCAM_RETURN_ERROR_PARAM,
        "soft image file(%s) write buf failed, buf is not valid", XCAM_STR (get_file_name ()));

    XCAM_ASSERT (is_valid ());
    uint32_t height = buf->get_height ();
    uint32_t line_bytes = buf->get_width () * buf->pixel_size ();

    for (uint32_t index = 0; index < height; index++) {
        uint8_t *line_ptr = buf->get_buf_ptr (0, index);
        XCAM_FAIL_RETURN (
            WARNING, fwrite (line_ptr, 1, line_bytes, _fp) == line_bytes, XCAM_RETURN_ERROR_FILE,
            "soft image file(%s) write buf failed, image_line:%d", XCAM_STR (get_file_name ()), index);
    }
    return XCAM_RETURN_NO_ERROR;
}

template <typename T> template <typename O>
O
SoftImage<T>::read_interpolate_data (float x, float y) const
{
    int32_t x0 = (int32_t)(x), y0 = (int32_t)(y);
    float a = x - x0, b = y - y0;
    O l0[2], l1[2];
    read_array<O, 2> (x0, y0, l0);
    read_array<O, 2> (x0, y0 + 1, l1);

    return l1[1] * (a * b) + l0[0] * ((1 - a) * (1 - b)) +
           l1[0] * ((1 - a) * b) + l0[1] * (a * (1 - b));
}

template <typename T> template<typename O, uint32_t N>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, O *array) const
{
    for (uint32_t i = 0; i < N; ++i) {
        array[i] = read_interpolate_data<O> (pos[i].x, pos[i].y);
    }
}

#if ENABLE_AVX512

// interpolate 8 pixels position
// result = p00 * (1 - weight.x) * (1 - weight.y) +
//          p01 * weight.x * (1 - weight.y) +
//          p10 * (1 - weight.x) * weight.y +
//          p11 * weight.x * weight.y;
template <typename T>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, Float2 *array) const
{
    Float2* dest = array;

    __m512 const_one = _mm512_set1_ps (1.0f);
    __m512 const_two = _mm512_set1_ps (2.0f);
    __m512i const_left_idx = _mm512_setr_epi32 (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    __m512i const_right_idx = _mm512_setr_epi32 (2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);
    __m512i const_shuffle = _mm512_setr_epi32 (0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);

    for (uint32_t i = 0; i < XCAM_SOFT_WORKUNIT_PIXELS / 8; i++) {
        __m512 interp_pos = _mm512_loadu_ps (&pos[8 * i]);
        __m512 interp_pos_xy = _mm512_floor_ps (interp_pos);
        __m512 interp_weight = _mm512_sub_ps (interp_pos, interp_pos_xy);
        __m512 weight_x = _mm512_permute_ps (interp_weight, 0xA0); //0b10100000
        __m512 weight_y = _mm512_permute_ps (interp_weight, 0xF5); //0b11110101
        __m512 weight_x_1 = _mm512_sub_ps(const_one, weight_x);
        __m512 weight_y_1 = _mm512_sub_ps(const_one, weight_y);

        int32_t pos_x0 = (int32_t)(pos[8 * i].x);
        border_check_x (pos_x0);

        float* pos_xy = (float*)&interp_pos_xy;
        __m512 pos_00 = _mm512_setr_ps (pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1],
                                        pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1],
                                        pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1],
                                        pos_xy[0], pos_xy[1], pos_xy[0], pos_xy[1]);

        __m512i pos_index = _mm512_cvtps_epu32 (_mm512_mul_ps (_mm512_sub_ps (interp_pos_xy, pos_00), const_two));
        __m512i idx_left = _mm512_add_epi32 (_mm512_permutexvar_epi32 (const_shuffle, pos_index), const_left_idx);
        __m512i idx_right = _mm512_add_epi32 (_mm512_permutexvar_epi32 (const_shuffle, pos_index), const_right_idx);

        int32_t pos_y0 = (int32_t)(pos[8 * i].y);
        border_check_y (pos_y0);
        const T* top_ptr = ((const T*)(_buf_ptr + pos_y0 * _pitch));

        int32_t pos_y1 = pos_y0 + 1;
        border_check_y (pos_y1);
        const T* bottom_ptr = ((const T*)(_buf_ptr + pos_y1 * _pitch));

        __m512 tl = _mm512_i32gather_ps (idx_left, (float*) & (top_ptr[pos_x0]), 4);
        __m512 tr = _mm512_i32gather_ps (idx_right, (float*) & (top_ptr[pos_x0]), 4);
        __m512 bl = _mm512_i32gather_ps (idx_left, (float*) & (bottom_ptr[pos_x0]), 4);
        __m512 br = _mm512_i32gather_ps (idx_right, (float*) & (bottom_ptr[pos_x0]), 4);

        __m512 interp_value = _mm512_fmadd_ps (tl, _mm512_mul_ps (weight_x_1, weight_y_1),
                                               _mm512_mul_ps (tr, _mm512_mul_ps (weight_x, weight_y_1))) +
                              _mm512_fmadd_ps (bl, _mm512_mul_ps (weight_x_1, weight_y),
                                               _mm512_mul_ps (br, _mm512_mul_ps (weight_x, weight_y)));

        _mm512_storeu_ps (dest, interp_value);
        dest += 8;
    }
}

template <typename T>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, Uchar *array, bool is_chroma) const
{
    float* dest = (float*)array;
    __m256 const_one_float = _mm256_set1_ps (1.0f);
    __m256i const_pitch = _mm256_set1_epi32 (_pitch);
    __m256i const_one_int = _mm256_set1_epi32 (1);
    uint32_t loop_count = XCAM_SOFT_WORKUNIT_PIXELS / 8;
    if (is_chroma) {
        loop_count = loop_count / 2;
    }

    for (uint32_t i = 0; i < loop_count; i++) {
        // load 8 interpolate pos ((8 x float2) x 32bit)
        __m512 interp_pos = _mm512_loadu_ps (&pos[8 * i]);
        __m512 interp_pos_xy = _mm512_floor_ps (interp_pos);
        __m512 interp_weight = _mm512_sub_ps (interp_pos, interp_pos_xy);
        __m256 weight_lo = _mm512_extractf32x8_ps (interp_weight, 0);
        __m256 weight_hi = _mm512_extractf32x8_ps (interp_weight, 1);

        __m256 weight_x = _mm256_shuffle_ps (weight_lo, weight_hi, 0x88);
        __m256 weight_y = _mm256_shuffle_ps (weight_lo, weight_hi, 0xDD);
        weight_x = (__m256)_mm256_permute4x64_epi64 ((__m256i)weight_x, 0xD8);
        weight_y = (__m256)_mm256_permute4x64_epi64 ((__m256i)weight_y, 0xD8);

        __m256 weight_x_1 = _mm256_sub_ps (const_one_float, weight_x);
        __m256 weight_y_1 = _mm256_sub_ps (const_one_float, weight_y);

        float* pos_xy = (float*)&interp_pos_xy;
        __m512 pos_00 = _mm512_setr_ps (0, pos_xy[1], 0, pos_xy[1],
                                        0, pos_xy[1], 0, pos_xy[1],
                                        0, pos_xy[1], 0, pos_xy[1],
                                        0, pos_xy[1], 0, pos_xy[1]);

        __m512i pos_index = _mm512_cvtps_epi32 (_mm512_sub_ps (interp_pos_xy, pos_00));
        // No two-source shuffle for _epi32 type, so we use _ps here.
        __m256 pos_idx_lo = _mm512_extractf32x8_ps ((__m512)pos_index, 0);
        __m256 pos_idx_hi = _mm512_extractf32x8_ps ((__m512)pos_index, 1);
        __m256i pos_idx_x = (__m256i)_mm256_shuffle_ps (pos_idx_lo, pos_idx_hi, 0x88);
        __m256i pos_idx_y = (__m256i)_mm256_shuffle_ps (pos_idx_lo, pos_idx_hi, 0xDD);
        pos_idx_x = _mm256_permute4x64_epi64 (pos_idx_x, 0xD8);
        pos_idx_y = _mm256_permute4x64_epi64 (pos_idx_y, 0xD8);

        int32_t pos_y0 = (int32_t)(pos[8 * i].y);
        border_check_y (pos_y0);
        const T* base = ((const T*)(_buf_ptr + pos_y0 * _pitch));

        __m256i offset_top0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (pos_idx_y, const_pitch));

        __m256i offset_top1 = _mm256_add_epi32 (offset_top0, const_one_int);

        __m256i offset_bottom0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (_mm256_add_epi32(pos_idx_y, const_one_int), const_pitch));

        __m256i offset_bottom1 = _mm256_add_epi32 (offset_bottom0, const_one_int);

        // this emulates gather of byte, the only thing we need to ensure
        // is that we should not read the last three-bytes of the buffer in-case we read out-of-bounds.
        // The reason is that we read four-bytes instead of one-byte here.
        __m256i mask = _mm256_set1_epi32 (0xFF);
        __m256i tmp = _mm256_i32gather_epi32 ((const int*)base, offset_top0, 1);
        tmp = _mm256_and_si256 (tmp, mask);
        __m256 pixel_tl = _mm256_cvtepi32_ps(tmp);

        tmp = _mm256_i32gather_epi32 ((const int*)base, offset_top1, 1);
        tmp = _mm256_and_si256 (tmp, mask);
        __m256 pixel_tr = _mm256_cvtepi32_ps(tmp);

        tmp = _mm256_i32gather_epi32 ((const int*)base, offset_bottom0, 1);
        tmp = _mm256_and_si256 (tmp, mask);
        __m256 pixel_bl = _mm256_cvtepi32_ps(tmp);

        tmp = _mm256_i32gather_epi32 ((const int*)base, offset_bottom1, 1);
        tmp = _mm256_and_si256 (tmp, mask);
        __m256 pixel_br = _mm256_cvtepi32_ps(tmp);

        __m256 interp_value_f = _mm256_fmadd_ps (pixel_tl, _mm256_mul_ps (weight_x_1, weight_y_1),
                                _mm256_mul_ps (pixel_tr, _mm256_mul_ps (weight_x, weight_y_1))) +
                                _mm256_fmadd_ps (pixel_bl, _mm256_mul_ps (weight_x_1, weight_y),
                                        _mm256_mul_ps (pixel_br, _mm256_mul_ps (weight_x, weight_y)));

        interp_value_f = _mm256_round_ps (interp_value_f, _MM_FROUND_TO_NEAREST_INT);
        __m256i interp_value = _mm256_cvtps_epi32 (interp_value_f);
        interp_value = _mm256_packs_epi32 (interp_value, interp_value);
        interp_value = _mm256_packus_epi16 (interp_value, interp_value);

        _mm_store_ss (dest, (__m128)_mm256_extractf128_si256 (interp_value, 0));
        _mm_store_ss (dest + 1, (__m128)_mm256_extractf128_si256 (interp_value, 1));
        dest += 2;
    }
}

static inline __m512 gather_uchar2x8(const int *base, __m256i offset)
{
    __m256i mask = _mm256_set1_epi32 (0xFF);
    __m256i value = _mm256_i32gather_epi32 (base, offset, 2);
    __m256i lo8 = _mm256_and_si256 (value, mask);
    __m256i shift8 = _mm256_srli_epi16 (value, 8);
    __m256i hi8 = _mm256_and_si256 (shift8, mask);
    __m256i tmp0 = _mm256_unpacklo_epi32 (lo8, hi8);
    __m256i tmp1 = _mm256_unpackhi_epi32 (lo8, hi8);
    __m256i tmp2 =  _mm256_permute2f128_si256 (tmp0, tmp1, 0x20);
    __m256i tmp3 =  _mm256_permute2f128_si256 (tmp0, tmp1, 0x31);

    __m512i tmp4 = _mm512_inserti64x4 (_mm512_castsi256_si512 (tmp2), tmp3, 1);

    return _mm512_cvtepi32_ps (tmp4);
}

template <typename T>
void
SoftImage<T>::read_interpolate_array (Float2 *pos, Uchar2 *array) const
{
    float* dest = (float*)array;

    __m512 const_one_float = _mm512_set1_ps (1.0f);
    __m256i const_pitch = _mm256_set1_epi32 (_pitch / 2);
    __m256i const_one_int = _mm256_set1_epi32 (1);
    __m256i index_x = _mm256_setr_epi32 (0, 2, 4, 6, 8, 10, 12, 14);
    __m256i index_y = _mm256_setr_epi32 (1, 3, 5, 7, 9, 11, 13, 15);

    // load 8 interpolate pos ((8 x float2) x 32bit)
    __m512 interp_pos = _mm512_loadu_ps (pos);

    __m512 interp_pos_xy = _mm512_floor_ps (interp_pos);
    __m512 interp_weight = _mm512_sub_ps (interp_pos, interp_pos_xy);

    __m512 weight_x = _mm512_permute_ps (interp_weight, 0xA0); //0b10100000
    __m512 weight_y = _mm512_permute_ps (interp_weight, 0xF5); //0b11110101
    __m512 weight_x_1 = _mm512_sub_ps(const_one_float, weight_x);
    __m512 weight_y_1 = _mm512_sub_ps(const_one_float, weight_y);

    float* pos_xy = (float*)&interp_pos_xy;
    __m512 pos_00 = _mm512_setr_ps (0, pos_xy[1], 0, pos_xy[1],
                                    0, pos_xy[1], 0, pos_xy[1],
                                    0, pos_xy[1], 0, pos_xy[1],
                                    0, pos_xy[1], 0, pos_xy[1]);

    __m512i pos_index = _mm512_cvtps_epi32 (_mm512_sub_ps (interp_pos_xy, pos_00));
    __m256i pos_idx_x = _mm256_i32gather_epi32 ((int32_t const*)(&pos_index), index_x, 4);
    __m256i pos_idx_y = _mm256_i32gather_epi32 ((int32_t const*)(&pos_index), index_y, 4);

    int32_t pos_y0 = (int32_t)(pos[0].y);
    border_check_y (pos_y0);

    int32_t pos_y1 = pos_y0 + 1;
    border_check_y (pos_y1);

    const T* base = ((const T*)(_buf_ptr + pos_y0 * _pitch));
    __m256i offset_top0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (pos_idx_y, const_pitch));

    __m256i offset_top1 = _mm256_add_epi32 (offset_top0, const_one_int);

    __m256i offset_bottom0 = _mm256_add_epi32 (pos_idx_x, _mm256_mullo_epi32 (_mm256_add_epi32(pos_idx_y, const_one_int), const_pitch));

    __m256i offset_bottom1 = _mm256_add_epi32 (offset_bottom0, const_one_int);

    __m512 pixel_tl = gather_uchar2x8((const int*)base, offset_top0);
    __m512 pixel_tr = gather_uchar2x8((const int*)base, offset_top1);
    __m512 pixel_bl = gather_uchar2x8((const int*)base, offset_bottom0);
    __m512 pixel_br = gather_uchar2x8((const int*)base, offset_bottom1);

    __m512 interp_value_f = _mm512_fmadd_ps (pixel_tl, _mm512_mul_ps (weight_x_1, weight_y_1),
                            _mm512_mul_ps (pixel_tr, _mm512_mul_ps (weight_x, weight_y_1))) +
                            _mm512_fmadd_ps (pixel_bl, _mm512_mul_ps (weight_x_1, weight_y),
                                    _mm512_mul_ps (pixel_br, _mm512_mul_ps (weight_x, weight_y)));

    //interp_value_f = _mm512_round_ps (interp_value_f, _MM_FROUND_TO_NEAREST_INT);

    __m512i interp_value = _mm512_cvtps_epi32 (interp_value_f);
    interp_value = _mm512_packs_epi32 (interp_value, interp_value);
    interp_value = _mm512_packus_epi16 (interp_value, interp_value);

    _mm_store_ss (dest,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 0));
    _mm_store_ss (dest + 1,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 1));
    _mm_store_ss (dest + 2,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 2));
    _mm_store_ss (dest + 3,  (__m128)_mm512_extractf32x4_ps ((__m512)interp_value, 3));
}

#endif

}
#endif //XCAM_SOFT_IMAGE_H
