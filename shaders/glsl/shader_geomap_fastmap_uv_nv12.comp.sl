#version 310 es

layout (local_size_x = 4, local_size_y = 4) in;

layout (binding = 0) readonly buffer InBufUV {
    uint data[];
} in_buf_uv;

layout (binding = 1) writeonly buffer OutBufUV {
    uvec2 data[];
} out_buf_uv;

layout (binding = 2) readonly buffer CoordX {
    vec4 data[];
} coordx;

layout (binding = 3) readonly buffer CoordY {
    vec4 data[];
} coordy;

uniform uint in_img_width;
uniform uint out_img_width;
uniform uint extended_offset;

uniform uint coords_width;

#define UNIT_SIZE 4u

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
    uvec4 x00_fract = (x00 % UNIT_SIZE) >> 1u;
    uvec4 x01_fract = (x01 % UNIT_SIZE) >> 1u;
    uvec4 x10_fract = (x10 % UNIT_SIZE) >> 1u;
    uvec4 x11_fract = (x11 % UNIT_SIZE) >> 1u;

    uvec4 index00 = y00 * in_img_width + x00_floor;
    uvec4 index01 = y01 * in_img_width + x01_floor;
    uvec4 index10 = y10 * in_img_width + x10_floor;
    uvec4 index11 = y11 * in_img_width + x11_floor;

    // pixel UV-value
    vec4 out00, out01, out10, out11;
    vec4 value = unpackUnorm4x8 (in_buf_uv.data[index00.x]);
    out00.xy = (x00_fract.x == 0u) ? value.xy : value.zw;
    value = (index01.x == index00.x) ? value : unpackUnorm4x8 (in_buf_uv.data[index01.x]);
    out01.xy = (x01_fract.x == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.x]);
    out10.xy = (x10_fract.x == 0u) ? value.xy : value.zw;
    value = (index11.x == index10.x) ? value : unpackUnorm4x8 (in_buf_uv.data[index11.x]);
    out11.xy = (x11_fract.x == 0u) ? value.xy : value.zw;

    value = unpackUnorm4x8 (in_buf_uv.data[index00.y]);
    out00.zw = (x00_fract.y == 0u) ? value.xy : value.zw;
    value = (index01.y == index00.y) ? value : unpackUnorm4x8 (in_buf_uv.data[index01.y]);
    out01.zw = (x01_fract.y == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.y]);
    out10.zw = (x10_fract.y == 0u) ? value.xy : value.zw;
    value = (index11.y == index10.y) ? value : unpackUnorm4x8 (in_buf_uv.data[index11.y]);
    out11.zw = (x11_fract.y == 0u) ? value.xy : value.zw;

    vec4 inter0 = out00 * weight00.xxyy + out01 * weight01.xxyy +
                  out10 * weight10.xxyy + out11 * weight11.xxyy;

    value = unpackUnorm4x8 (in_buf_uv.data[index00.z]);
    out00.xy = (x00_fract.z == 0u) ? value.xy : value.zw;
    value = (index01.z == index00.z) ? value : unpackUnorm4x8 (in_buf_uv.data[index01.z]);
    out01.xy = (x01_fract.z == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.z]);
    out10.xy = (x10_fract.z == 0u) ? value.xy : value.zw;
    value = (index11.z == index10.z) ? value : unpackUnorm4x8 (in_buf_uv.data[index11.z]);
    out11.xy = (x11_fract.z == 0u) ? value.xy : value.zw;

    value = unpackUnorm4x8 (in_buf_uv.data[index00.w]);
    out00.zw = (x00_fract.w == 0u) ? value.xy : value.zw;
    value = (index01.w == index00.w) ? value : unpackUnorm4x8 (in_buf_uv.data[index01.w]);
    out01.zw = (x01_fract.w == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.w]);
    out10.zw = (x10_fract.w == 0u) ? value.xy : value.zw;
    value = (index11.w == index10.w) ? value : unpackUnorm4x8 (in_buf_uv.data[index11.w]);
    out11.zw = (x11_fract.w == 0u) ? value.xy : value.zw;
    vec4 inter1 = out00 * weight00.zzww + out01 * weight01.zzww +
                  out10 * weight10.zzww + out11 * weight11.zzww;

    uint out_pos = gl_GlobalInvocationID.y * out_img_width + extended_offset + gl_GlobalInvocationID.x;
    out_buf_uv.data[out_pos] = uvec2 (packUnorm4x8 (inter0), packUnorm4x8 (inter1));
}
