#version 310 es

layout (local_size_x = 8, local_size_y = 8) in;

layout (binding = 0) readonly buffer InBufY {
    uvec4 data[];
} in_buf_y;

layout (binding = 1) readonly buffer InBufU {
    uvec2 data[];
} in_buf_u;

layout (binding = 2) readonly buffer InBufV {
    uvec2 data[];
} in_buf_v;

layout (binding = 3) writeonly buffer OutBufY {
    uvec2 data[];
} out_buf_y;

layout (binding = 4) writeonly buffer OutBufU {
    uint data[];
} out_buf_u;

layout (binding = 5) writeonly buffer OutBufV {
    uint data[];
} out_buf_v;

uniform uint in_img_width;
uniform uint in_img_height;
uniform uint in_offset_x;

uniform uint out_img_width;
uniform uint merge_width;

const float coeffs[5] = float[] (0.152f, 0.222f, 0.252f, 0.222f, 0.152f);

#define unpack_unorm_y(buf, pixel, idx) \
    { \
        uvec4 value = buf.data[idx]; \
        pixel[0] = unpackUnorm4x8 (value.w); \
        value = buf.data[idx + 1u]; \
        pixel[1] = unpackUnorm4x8 (value.x); \
        pixel[2] = unpackUnorm4x8 (value.y); \
        pixel[3] = unpackUnorm4x8 (value.z); \
        pixel[4] = unpackUnorm4x8 (value.w); \
        value = buf.data[idx + 2u]; \
        pixel[5] = unpackUnorm4x8 (value.x); \
    }

#define multiply_coeff_y(sum, pixel, idx) \
    { \
        sum[0] += pixel[0] * coeffs[idx]; \
        sum[1] += pixel[1] * coeffs[idx]; \
        sum[2] += pixel[2] * coeffs[idx]; \
        sum[3] += pixel[3] * coeffs[idx]; \
        sum[4] += pixel[4] * coeffs[idx]; \
        sum[5] += pixel[5] * coeffs[idx]; \
    }

#define unpack_unorm_uv(buf, pixel, idx) \
    { \
        uvec2 value = buf.data[idx]; \
        pixel[0] = unpackUnorm4x8 (value.y); \
        value = buf.data[idx + 1u]; \
        pixel[1] = unpackUnorm4x8 (value.x); \
        pixel[2] = unpackUnorm4x8 (value.y); \
        value = buf.data[idx + 2u]; \
        pixel[3] = unpackUnorm4x8 (value.x); \
    }

#define multiply_coeff_uv(sum, pixel, idx) \
    { \
        sum[0] += pixel[0] * coeffs[idx]; \
        sum[1] += pixel[1] * coeffs[idx]; \
        sum[2] += pixel[2] * coeffs[idx]; \
        sum[3] += pixel[3] * coeffs[idx]; \
    }

void gauss_scale_y (uvec2 y_id);
void gauss_scale_uv (uvec2 uv_id);

void main ()
{
    uvec2 g_id = gl_GlobalInvocationID.xy;

    uvec2 y_id = uvec2 (g_id.x, g_id.y * 2u);
    gauss_scale_y (y_id);

    gauss_scale_uv (g_id);
}

void gauss_scale_y (uvec2 y_id)
{
    uvec2 in_id = uvec2 (y_id.x, y_id.y * 2u);
    uvec2 gauss_start = in_id - uvec2 (1u, 2u);
    gauss_start.y = clamp (gauss_start.y, 0u, in_img_height - 7u);

    vec4 sum0[6] = vec4[] (vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f));
    vec4 sum1[6] = vec4[] (vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f));

    vec4 pixel_y[6];
    uint in_idx = (in_id.y == 0u) ? (in_id.x - 1u) : (gauss_start.y * in_img_width + gauss_start.x);
    in_idx += in_offset_x;
    unpack_unorm_y (in_buf_y, pixel_y, in_idx);
    multiply_coeff_y (sum0, pixel_y, 0u);

    in_idx = (in_id.y == 0u) ? in_idx : (in_idx + in_img_width);
    unpack_unorm_y (in_buf_y, pixel_y, in_idx);
    multiply_coeff_y (sum0, pixel_y, 1u);

    in_idx = (in_id.y == 0u) ? in_idx : (in_idx + in_img_width);
    unpack_unorm_y (in_buf_y, pixel_y, in_idx);
    multiply_coeff_y (sum0, pixel_y, 2u);
    multiply_coeff_y (sum1, pixel_y, 0u);

    in_idx += in_img_width;
    unpack_unorm_y (in_buf_y, pixel_y, in_idx);
    multiply_coeff_y (sum0, pixel_y, 3u);
    multiply_coeff_y (sum1, pixel_y, 1u);

    in_idx += in_img_width;
    unpack_unorm_y (in_buf_y, pixel_y, in_idx);
    multiply_coeff_y (sum0, pixel_y, 4u);
    multiply_coeff_y (sum1, pixel_y, 2u);

    in_idx += in_img_width;
    unpack_unorm_y (in_buf_y, pixel_y, in_idx);
    multiply_coeff_y (sum1, pixel_y, 3u);

    in_idx += in_img_width;
    unpack_unorm_y (in_buf_y, pixel_y, in_idx);
    multiply_coeff_y (sum1, pixel_y, 4u);

    sum0[0] = (in_id.x == 0u) ? sum0[1].xxxx : sum0[0];
    sum0[5] = (in_id.x >= merge_width - 2u) ? sum0[4].wwww : sum0[5];
    vec4 out_data00 =
        vec4 (sum0[0].z, sum0[1].x, sum0[1].z, sum0[2].x) * coeffs[0] +
        vec4 (sum0[0].w, sum0[1].y, sum0[1].w, sum0[2].y) * coeffs[1] +
        vec4 (sum0[1].x, sum0[1].z, sum0[2].x, sum0[2].z) * coeffs[2] +
        vec4 (sum0[1].y, sum0[1].w, sum0[2].y, sum0[2].w) * coeffs[3] +
        vec4 (sum0[1].z, sum0[2].x, sum0[2].z, sum0[3].x) * coeffs[4];
    out_data00 = clamp (out_data00, 0.0f, 1.0f);

    sum1[0] = (in_id.x == 0u) ? sum1[1].xxxx : sum1[0];
    sum1[5] = (in_id.x >= merge_width - 2u) ? sum1[4].wwww : sum1[5];
    vec4 out_data01 =
        vec4 (sum0[2].z, sum0[3].x, sum0[3].z, sum0[4].x) * coeffs[0] +
        vec4 (sum0[2].w, sum0[3].y, sum0[3].w, sum0[4].y) * coeffs[1] +
        vec4 (sum0[3].x, sum0[3].z, sum0[4].x, sum0[4].z) * coeffs[2] +
        vec4 (sum0[3].y, sum0[3].w, sum0[4].y, sum0[4].w) * coeffs[3] +
        vec4 (sum0[3].z, sum0[4].x, sum0[4].z, sum0[5].x) * coeffs[4];
    out_data01 = clamp (out_data01, 0.0f, 1.0f);

    y_id.x = clamp (y_id.x, 0u, out_img_width - 1u);
    uint out_idx = y_id.y * out_img_width + y_id.x;
    out_buf_y.data[out_idx] = uvec2 (packUnorm4x8 (out_data00), packUnorm4x8 (out_data01));

    vec4 out_data10 =
        vec4 (sum1[0].z, sum1[1].x, sum1[1].z, sum1[2].x) * coeffs[0] +
        vec4 (sum1[0].w, sum1[1].y, sum1[1].w, sum1[2].y) * coeffs[1] +
        vec4 (sum1[1].x, sum1[1].z, sum1[2].x, sum1[2].z) * coeffs[2] +
        vec4 (sum1[1].y, sum1[1].w, sum1[2].y, sum1[2].w) * coeffs[3] +
        vec4 (sum1[1].z, sum1[2].x, sum1[2].z, sum1[3].x) * coeffs[4];
    out_data10 = clamp (out_data10, 0.0f, 1.0f);

    vec4 out_data11 =
        vec4 (sum1[2].z, sum1[3].x, sum1[3].z, sum1[4].x) * coeffs[0] +
        vec4 (sum1[2].w, sum1[3].y, sum1[3].w, sum1[4].y) * coeffs[1] +
        vec4 (sum1[3].x, sum1[3].z, sum1[4].x, sum1[4].z) * coeffs[2] +
        vec4 (sum1[3].y, sum1[3].w, sum1[4].y, sum1[4].w) * coeffs[3] +
        vec4 (sum1[3].z, sum1[4].x, sum1[4].z, sum1[5].x) * coeffs[4];
    out_data11 = clamp (out_data11, 0.0f, 1.0f);

    out_buf_y.data[out_idx + out_img_width] = uvec2 (packUnorm4x8 (out_data10), packUnorm4x8 (out_data11));
}

void gauss_scale_uv (uvec2 uv_id)
{
    uvec2 in_id = uvec2 (uv_id.x, uv_id.y * 2u);
    uvec2 gauss_start = in_id - uvec2 (1u, 2u);
    gauss_start.y = clamp (gauss_start.y, 0u, (in_img_height >> 1u) - 5u);

    vec4 sum_u[4] = vec4[] (vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f));
    vec4 sum_v[4] = vec4[] (vec4 (0.0f), vec4 (0.0f), vec4 (0.0f), vec4 (0.0f));
    uint in_idx = (in_id.y == 0u) ? (in_id.x - 1u) : (gauss_start.y * in_img_width + gauss_start.x);
    in_idx += in_offset_x;

    vec4 pixel_u[4], pixel_v[4];
    unpack_unorm_uv (in_buf_u, pixel_u, in_idx);
    multiply_coeff_uv (sum_u, pixel_u, 0u);
    unpack_unorm_uv (in_buf_v, pixel_v, in_idx);
    multiply_coeff_uv (sum_v, pixel_v, 0u);

    in_idx = (in_id.y == 0u) ? in_idx : (in_idx + in_img_width);
    unpack_unorm_uv (in_buf_u, pixel_u, in_idx);
    multiply_coeff_uv (sum_u, pixel_u, 1u);
    unpack_unorm_uv (in_buf_v, pixel_v, in_idx);
    multiply_coeff_uv (sum_v, pixel_v, 1u);

    in_idx = (in_id.y == 0u) ? in_idx : (in_idx + in_img_width);
    unpack_unorm_uv (in_buf_u, pixel_u, in_idx);
    multiply_coeff_uv (sum_u, pixel_u, 2u);
    unpack_unorm_uv (in_buf_v, pixel_v, in_idx);
    multiply_coeff_uv (sum_v, pixel_v, 2u);

    in_idx += in_img_width;
    unpack_unorm_uv (in_buf_u, pixel_u, in_idx);
    multiply_coeff_uv (sum_u, pixel_u, 3u);
    unpack_unorm_uv (in_buf_v, pixel_v, in_idx);
    multiply_coeff_uv (sum_v, pixel_v, 3u);

    in_idx += in_img_width;
    unpack_unorm_uv (in_buf_u, pixel_u, in_idx);
    multiply_coeff_uv (sum_u, pixel_u, 4u);
    unpack_unorm_uv (in_buf_v, pixel_v, in_idx);
    multiply_coeff_uv (sum_v, pixel_v, 4u);

    sum_u[0] = (in_id.x == 0u) ? sum_u[1].xxxx : sum_u[0];
    sum_u[3] = (in_id.x >= merge_width - 2u) ? sum_u[2].wwww: sum_u[3];
    vec4 out_data =
        vec4 (sum_u[0].z, sum_u[1].x, sum_u[1].z, sum_u[2].x) * coeffs[0] +
        vec4 (sum_u[0].w, sum_u[1].y, sum_u[1].w, sum_u[2].y) * coeffs[1] +
        vec4 (sum_u[1].x, sum_u[1].z, sum_u[2].x, sum_u[2].z) * coeffs[2] +
        vec4 (sum_u[1].y, sum_u[1].w, sum_u[2].y, sum_u[2].w) * coeffs[3] +
        vec4 (sum_u[1].z, sum_u[2].x, sum_u[2].z, sum_u[3].x) * coeffs[4];
    out_data = clamp (out_data, 0.0f, 1.0f);

    uv_id.x = clamp (uv_id.x, 0u, out_img_width - 1u);
    uint out_idx = uv_id.y * out_img_width + uv_id.x;
    out_buf_u.data[out_idx] = packUnorm4x8 (out_data);

    sum_v[0] = (in_id.x == 0u) ? sum_v[1].xxxx : sum_v[0];
    sum_v[3] = (in_id.x >= merge_width - 2u) ? sum_v[2].wwww : sum_v[3];
    out_data =
        vec4 (sum_v[0].z, sum_v[1].x, sum_v[1].z, sum_v[2].x) * coeffs[0] +
        vec4 (sum_v[0].w, sum_v[1].y, sum_v[1].w, sum_v[2].y) * coeffs[1] +
        vec4 (sum_v[1].x, sum_v[1].z, sum_v[2].x, sum_v[2].z) * coeffs[2] +
        vec4 (sum_v[1].y, sum_v[1].w, sum_v[2].y, sum_v[2].w) * coeffs[3] +
        vec4 (sum_v[1].z, sum_v[2].x, sum_v[2].z, sum_v[3].x) * coeffs[4];
    out_data = clamp (out_data, 0.0f, 1.0f);

    out_buf_v.data[out_idx] = packUnorm4x8 (out_data);
}
