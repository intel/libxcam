#version 310 es

layout (local_size_x = 4, local_size_y = 4) in;

layout (binding = 0) readonly buffer InBufU {
    uint data[];
} in_buf_u;

layout (binding = 1) readonly buffer InBufV {
    uint data[];
} in_buf_v;

layout (binding = 2) writeonly buffer OutBufU {
    uint data[];
} out_buf_u;

layout (binding = 3) writeonly buffer OutBufV {
    uint data[];
} out_buf_v;

layout (binding = 4) readonly buffer CoordX {
    vec4 data[];
} coordx;

layout (binding = 5) readonly buffer CoordY {
    vec4 data[];
} coordy;

uniform uint in_img_width;
uniform uint out_img_width;
uniform uint extended_offset;

uniform uint coords_width;

#define UNIT_SIZE 4u

#define unpack_unorm(buf, index) \
    { \
        vec4 value = unpackUnorm4x8 (buf.data[index00[index]]); \
        out00[index] = value[x00_fract[index]]; \
        value = (index01[index] == index00[index]) ? value : unpackUnorm4x8 (buf.data[index01[index]]); \
        out01[index] = value[x01_fract[index]]; \
        value = unpackUnorm4x8 (buf.data[index10[index]]); \
        out10[index] = value[x10_fract[index]]; \
        value = (index11[index] == index10[index]) ? value : unpackUnorm4x8 (buf.data[index11[index]]); \
        out11[index] = value[x11_fract[index]]; \
    }

void main ()
{
    uint coord_pos = gl_GlobalInvocationID.y * coords_width + gl_GlobalInvocationID.x;
    vec4 in_uv_x = coordx.data[coord_pos];
    vec4 in_uv_y = coordy.data[coord_pos];

    uvec4 x00 = uvec4 (in_uv_x);
    uvec4 y00 = uvec4 (in_uv_y);
    uvec4 x01 = x00 + 1u;
    uvec4 y01 = y00;
    uvec4 x10 = x00;
    uvec4 y10 = y00 + 1u;
    uvec4 x11 = x01;
    uvec4 y11 = y10;

    vec4 fract_x = fract (in_uv_x);
    vec4 fract_y = fract (in_uv_y);
    vec4 weight00 = (1.0f - fract_x) * (1.0f - fract_y);
    vec4 weight01 = fract_x * (1.0f - fract_y);
    vec4 weight10 = (1.0f - fract_x) * fract_y;
    vec4 weight11 = fract_x * fract_y;

    uvec4 x00_floor = x00 >> 2u;    // divided by UNIT_SIZE
    uvec4 x01_floor = x01 >> 2u;
    uvec4 x10_floor = x10 >> 2u;
    uvec4 x11_floor = x11 >> 2u;
    uvec4 x00_fract = x00 % UNIT_SIZE;
    uvec4 x01_fract = x01 % UNIT_SIZE;
    uvec4 x10_fract = x10 % UNIT_SIZE;
    uvec4 x11_fract = x11 % UNIT_SIZE;

    uvec4 index00 = y00 * in_img_width + x00_floor;
    uvec4 index01 = y01 * in_img_width + x01_floor;
    uvec4 index10 = y10 * in_img_width + x10_floor;
    uvec4 index11 = y11 * in_img_width + x11_floor;

    // pixel U-value
    vec4 out00, out01, out10, out11;
    unpack_unorm (in_buf_u, 0);
    unpack_unorm (in_buf_u, 1);
    unpack_unorm (in_buf_u, 2);
    unpack_unorm (in_buf_u, 3);
    vec4 inter = out00 * weight00 + out01 * weight01 + out10 * weight10 + out11 * weight11;
    uint out_pos = gl_GlobalInvocationID.y * out_img_width + extended_offset + gl_GlobalInvocationID.x;
    out_buf_u.data[out_pos] = packUnorm4x8 (inter);

    // pixel V-value
    unpack_unorm (in_buf_v, 0);
    unpack_unorm (in_buf_v, 1);
    unpack_unorm (in_buf_v, 2);
    unpack_unorm (in_buf_v, 3);
    inter = out00 * weight00 + out01 * weight01 + out10 * weight10 + out11 * weight11;
    out_buf_v.data[out_pos] = packUnorm4x8 (inter);
}
