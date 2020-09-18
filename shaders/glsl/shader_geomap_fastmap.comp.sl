#version 310 es

layout (local_size_x = 4, local_size_y = 4) in;

layout (binding = 0) readonly buffer InBufY {
    uint data[];
} in_buf_y;

layout (binding = 1) readonly buffer InBufUV {
    uint data[];
} in_buf_uv;

layout (binding = 2) writeonly buffer OutBufY {
    uint data[];
} out_buf_y;

layout (binding = 3) writeonly buffer OutBufUV {
    uint data[];
} out_buf_uv;

layout (binding = 4) readonly buffer CoordX {
    vec4 data[];
} coordx;

layout (binding = 5) readonly buffer CoordY {
    vec4 data[];
} coordy;

uniform uint in_img_width;
uniform uint in_img_height;

uniform uint out_img_width;
uniform uint extended_offset;

uniform uint coords_width;

#define UNIT_SIZE 4u

#define unpack_unorm_y(index) \
    { \
        vec4 value = unpackUnorm4x8 (in_buf_y.data[index00[index]]); \
        out_y00[index] = value[x00_fract[index]]; \
        value = (index01[index] == index00[index]) ? value : unpackUnorm4x8 (in_buf_y.data[index01[index]]); \
        out_y01[index] = value[x01_fract[index]]; \
        value = unpackUnorm4x8 (in_buf_y.data[index10[index]]); \
        out_y10[index] = value[x10_fract[index]]; \
        value = (index11[index] == index10[index]) ? value : unpackUnorm4x8 (in_buf_y.data[index11[index]]); \
        out_y11[index] = value[x11_fract[index]]; \
    }

void geomap_y (vec4 in_img_x, vec4 in_img_y, out uint out_data);
void geomap_uv (vec2 in_uv_x, vec2 in_uv_y, out uint out_data);

void main ()
{
    uint g_x = gl_GlobalInvocationID.x;
    uint g_y = gl_GlobalInvocationID.y << 1u;

    uint out_x = g_x + extended_offset;
    uint out_pos = g_y * out_img_width + out_x;
    uint coord_pos = g_y * coords_width + g_x;

    uint out_data;
    vec4 in_img_x = coordx.data[coord_pos];
    vec4 in_img_y = coordy.data[coord_pos];
    geomap_y (in_img_x, in_img_y, out_data);
    out_buf_y.data[out_pos] = out_data;

    vec2 in_uv_x = in_img_x.xz;
    vec2 in_uv_y = in_img_y.xz / 2.0f;
    in_uv_y = clamp (in_uv_y, 0.0f, float ((in_img_height >> 1u) - 1u));
    geomap_uv (in_uv_x, in_uv_y, out_data);
    out_buf_uv.data[(g_y >> 1u) * out_img_width + out_x] = out_data;

    coord_pos += coords_width;
    in_img_x = coordx.data[coord_pos];
    in_img_y = coordy.data[coord_pos];
    geomap_y (in_img_x, in_img_y, out_data);
    out_buf_y.data[out_pos + out_img_width] = out_data;
}

void geomap_y (vec4 in_img_x, vec4 in_img_y, out uint out_data)
{
    uvec4 x00 = uvec4 (in_img_x);
    uvec4 y00 = uvec4 (in_img_y);
    uvec4 x01 = x00 + 1u;
    uvec4 y01 = y00;
    uvec4 x10 = x00;
    uvec4 y10 = y00 + 1u;
    uvec4 x11 = x01;
    uvec4 y11 = y10;

    vec4 fract_x = fract (in_img_x);
    vec4 fract_y = fract (in_img_y);
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

    // pixel Y-value
    vec4 out_y00, out_y01, out_y10, out_y11;
    unpack_unorm_y (0);
    unpack_unorm_y (1);
    unpack_unorm_y (2);
    unpack_unorm_y (3);

    vec4 inter_y = out_y00 * weight00 + out_y01 * weight01 + out_y10 * weight10 + out_y11 * weight11;
    out_data = packUnorm4x8 (inter_y);
}

void geomap_uv (vec2 in_uv_x, vec2 in_uv_y, out uint out_data)
{
    uvec2 x00 = uvec2 (in_uv_x);
    uvec2 y00 = uvec2 (in_uv_y);
    uvec2 x01 = x00 + 1u;
    uvec2 y01 = y00;
    uvec2 x10 = x00;
    uvec2 y10 = y00 + 1u;
    uvec2 x11 = x01;
    uvec2 y11 = y10;

    vec2 fract_x = fract (in_uv_x);
    vec2 fract_y = fract (in_uv_y);
    vec2 weight00 = (1.0f - fract_x) * (1.0f - fract_y);
    vec2 weight01 = fract_x * (1.0f - fract_y);
    vec2 weight10 = (1.0f - fract_x) * fract_y;
    vec2 weight11 = fract_x * fract_y;

    uvec2 x00_floor = x00 >> 2u;    // divided by UNIT_SIZE
    uvec2 x01_floor = x01 >> 2u;
    uvec2 x10_floor = x10 >> 2u;
    uvec2 x11_floor = x11 >> 2u;
    uvec2 x00_fract = (x00 % UNIT_SIZE) >> 1u;
    uvec2 x01_fract = (x01 % UNIT_SIZE) >> 1u;
    uvec2 x10_fract = (x10 % UNIT_SIZE) >> 1u;
    uvec2 x11_fract = (x11 % UNIT_SIZE) >> 1u;

    uvec2 index00 = y00 * in_img_width + x00_floor;
    uvec2 index01 = y01 * in_img_width + x01_floor;
    uvec2 index10 = y10 * in_img_width + x10_floor;
    uvec2 index11 = y11 * in_img_width + x11_floor;

    // pixel UV-value
    vec4 out_uv00, out_uv01, out_uv10, out_uv11;
    vec4 value = unpackUnorm4x8 (in_buf_uv.data[index00.x]);
    out_uv00.xy = (x00_fract.x == 0u) ? value.xy : value.zw;
    value = (index01.x == index00.x) ? value : unpackUnorm4x8 (in_buf_uv.data[index01.x]);
    out_uv01.xy = (x01_fract.x == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.x]);
    out_uv10.xy = (x10_fract.x == 0u) ? value.xy : value.zw;
    value = (index11.x == index10.x) ? value : unpackUnorm4x8 (in_buf_uv.data[index11.x]);
    out_uv11.xy = (x11_fract.x == 0u) ? value.xy : value.zw;

    value = unpackUnorm4x8 (in_buf_uv.data[index00.y]);
    out_uv00.zw = (x00_fract.y == 0u) ? value.xy : value.zw;
    value = (index01.y == index00.y) ? value : unpackUnorm4x8 (in_buf_uv.data[index01.y]);
    out_uv01.zw = (x01_fract.y == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv.data[index10.y]);
    out_uv10.zw = (x10_fract.y == 0u) ? value.xy : value.zw;
    value = (index11.y == index10.y) ? value : unpackUnorm4x8 (in_buf_uv.data[index11.y]);
    out_uv11.zw = (x11_fract.y == 0u) ? value.xy : value.zw;

    vec4 inter_uv = out_uv00 * weight00.xxyy + out_uv01 * weight01.xxyy +
                    out_uv10 * weight10.xxyy + out_uv11 * weight11.xxyy;
    out_data = packUnorm4x8 (inter_uv);
}
