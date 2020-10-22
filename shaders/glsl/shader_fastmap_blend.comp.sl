#version 310 es

layout (local_size_x = 4, local_size_y = 4) in;

layout (binding = 0) readonly buffer InBufY0 {
    uint data[];
} in_buf_y0;

layout (binding = 1) readonly buffer InBufUV0 {
    uint data[];
} in_buf_uv0;

layout (binding = 2) readonly buffer CoordX0 {
    vec4 data[];
} coordx0;

layout (binding = 3) readonly buffer CoordY0 {
    vec4 data[];
} coordy0;

layout (binding = 4) readonly buffer InBufY1 {
    uint data[];
} in_buf_y1;

layout (binding = 5) readonly buffer InBufUV1 {
    uint data[];
} in_buf_uv1;

layout (binding = 6) readonly buffer CoordX1 {
    vec4 data[];
} coordx1;

layout (binding = 7) readonly buffer CoordY1 {
    vec4 data[];
} coordy1;

layout (binding = 8) writeonly buffer OutBufY {
    uint data[];
} out_buf_y;

layout (binding = 9) writeonly buffer OutBufUV {
    uint data[];
} out_buf_uv;

layout (binding = 10) readonly buffer MaskBuf {
    uint data[];
} mask_buf;

uniform uint in_img_width0;
uniform uint in_img_width1;
uniform uint in_img_height;

uniform uint out_img_width;
uniform uint out_offset_x;

uniform uint coords_width0;
uniform uint coords_width1;

#define UNIT_SIZE 4u

void geomap_y0 (vec4 in_img_x, vec4 in_img_y, out vec4 out_data);
void geomap_y1 (vec4 in_img_x, vec4 in_img_y, out vec4 out_data);
void geomap_uv (vec4 in_uv_x, vec4 in_uv_y, out vec4 out_data0, out vec4 out_data1);

void main ()
{
    uint g_x = gl_GlobalInvocationID.x;
    uint g_y = gl_GlobalInvocationID.y << 1u;

    uint out_x = g_x + out_offset_x;
    uint out_pos = g_y * out_img_width + out_x;
    uvec2 coord_pos = g_y *  uvec2 (coords_width0, coords_width1) + g_x;

    vec4 out_data0;
    vec4 in_img_x0 = coordx0.data[coord_pos.x];
    vec4 in_img_y0 = coordy0.data[coord_pos.x];
    geomap_y0 (in_img_x0, in_img_y0, out_data0);

    vec4 out_data1;
    vec4 in_img_x1 = coordx1.data[coord_pos.y];
    vec4 in_img_y1 = coordy1.data[coord_pos.y];
    geomap_y1 (in_img_x1, in_img_y1, out_data1);

    vec4 mask = unpackUnorm4x8 (mask_buf.data[g_x]);
    vec4 out_y = (out_data0 - out_data1) * mask + out_data1;
    out_y = clamp (out_y, 0.0f, 1.0f);
    out_buf_y.data[out_pos] = packUnorm4x8 (out_y);

    vec4 in_uv_x = vec4 (in_img_x0.xz, in_img_x1.xz);
    vec4 in_uv_y = vec4 (in_img_y0.xz, in_img_y1.xz) / 2.0f;
    in_uv_y = clamp (in_uv_y, 0.0f, float ((in_img_height >> 1u) - 1u));
    geomap_uv (in_uv_x, in_uv_y, out_data0, out_data1);
    vec4 out_uv = (out_data0 - out_data1) * mask.xxzz + out_data1;
    out_uv = clamp (out_uv, 0.0f, 1.0f);
    out_buf_uv.data[(g_y >> 1u) * out_img_width + out_x] = packUnorm4x8 (out_uv);

    coord_pos += uvec2 (coords_width0, coords_width1);
    in_img_x0 = coordx0.data[coord_pos.x];
    in_img_y0 = coordy0.data[coord_pos.x];
    geomap_y0 (in_img_x0, in_img_y0, out_data0);

    in_img_x1 = coordx1.data[coord_pos.y];
    in_img_y1 = coordy1.data[coord_pos.y];
    geomap_y1 (in_img_x1, in_img_y1, out_data1);

    out_y = (out_data0 - out_data1) * mask + out_data1;
    out_y = clamp (out_y, 0.0f, 1.0f);
    out_buf_y.data[out_pos + out_img_width] = packUnorm4x8 (out_y);
}

#define unpack_unorm_y(index, in_buf_y) \
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

void geomap_y0 (vec4 in_img_x, vec4 in_img_y, out vec4 out_data)
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

    uvec4 index00 = y00 * in_img_width0 + x00_floor;
    uvec4 index01 = y01 * in_img_width0 + x01_floor;
    uvec4 index10 = y10 * in_img_width0 + x10_floor;
    uvec4 index11 = y11 * in_img_width0 + x11_floor;

    vec4 out_y00, out_y01, out_y10, out_y11;
    unpack_unorm_y (0, in_buf_y0);
    unpack_unorm_y (1, in_buf_y0);
    unpack_unorm_y (2, in_buf_y0);
    unpack_unorm_y (3, in_buf_y0);

    out_data = out_y00 * weight00 + out_y01 * weight01 + out_y10 * weight10 + out_y11 * weight11;
}

void geomap_y1 (vec4 in_img_x, vec4 in_img_y, out vec4 out_data)
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

    uvec4 index00 = y00 * in_img_width1 + x00_floor;
    uvec4 index01 = y01 * in_img_width1 + x01_floor;
    uvec4 index10 = y10 * in_img_width1 + x10_floor;
    uvec4 index11 = y11 * in_img_width1 + x11_floor;

    vec4 out_y00, out_y01, out_y10, out_y11;
    unpack_unorm_y (0, in_buf_y1);
    unpack_unorm_y (1, in_buf_y1);
    unpack_unorm_y (2, in_buf_y1);
    unpack_unorm_y (3, in_buf_y1);

    out_data = out_y00 * weight00 + out_y01 * weight01 + out_y10 * weight10 + out_y11 * weight11;
}

void geomap_uv (vec4 in_uv_x, vec4 in_uv_y, out vec4 out_data0, out vec4 out_data1)
{
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

    uvec4 img_width = uvec4 (in_img_width0, in_img_width0, in_img_width1, in_img_width1);
    uvec4 index00 = y00 * img_width + x00_floor;
    uvec4 index01 = y01 * img_width + x01_floor;
    uvec4 index10 = y10 * img_width + x10_floor;
    uvec4 index11 = y11 * img_width + x11_floor;

    vec4 out_uv00, out_uv01, out_uv10, out_uv11;
    vec4 value = unpackUnorm4x8 (in_buf_uv0.data[index00.x]);
    out_uv00.xy = (x00_fract.x == 0u) ? value.xy : value.zw;
    value = (index01.x == index00.x) ? value : unpackUnorm4x8 (in_buf_uv0.data[index01.x]);
    out_uv01.xy = (x01_fract.x == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv0.data[index10.x]);
    out_uv10.xy = (x10_fract.x == 0u) ? value.xy : value.zw;
    value = (index11.x == index10.x) ? value : unpackUnorm4x8 (in_buf_uv0.data[index11.x]);
    out_uv11.xy = (x11_fract.x == 0u) ? value.xy : value.zw;

    value = unpackUnorm4x8 (in_buf_uv0.data[index00.y]);
    out_uv00.zw = (x00_fract.y == 0u) ? value.xy : value.zw;
    value = (index01.y == index00.y) ? value : unpackUnorm4x8 (in_buf_uv0.data[index01.y]);
    out_uv01.zw = (x01_fract.y == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv0.data[index10.y]);
    out_uv10.zw = (x10_fract.y == 0u) ? value.xy : value.zw;
    value = (index11.y == index10.y) ? value : unpackUnorm4x8 (in_buf_uv0.data[index11.y]);
    out_uv11.zw = (x11_fract.y == 0u) ? value.xy : value.zw;

    out_data0 = out_uv00 * weight00.xxyy + out_uv01 * weight01.xxyy +
                out_uv10 * weight10.xxyy + out_uv11 * weight11.xxyy;

    value = unpackUnorm4x8 (in_buf_uv1.data[index00.z]);
    out_uv00.xy = (x00_fract.z == 0u) ? value.xy : value.zw;
    value = (index01.z == index00.z) ? value : unpackUnorm4x8 (in_buf_uv1.data[index01.z]);
    out_uv01.xy = (x01_fract.z == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv1.data[index10.z]);
    out_uv10.xy = (x10_fract.z == 0u) ? value.xy : value.zw;
    value = (index11.z == index10.z) ? value : unpackUnorm4x8 (in_buf_uv1.data[index11.z]);
    out_uv11.xy = (x11_fract.z == 0u) ? value.xy : value.zw;

    value = unpackUnorm4x8 (in_buf_uv1.data[index00.w]);
    out_uv00.zw = (x00_fract.w == 0u) ? value.xy : value.zw;
    value = (index01.w == index00.w) ? value : unpackUnorm4x8 (in_buf_uv1.data[index01.w]);
    out_uv01.zw = (x01_fract.w == 0u) ? value.xy : value.zw;
    value = unpackUnorm4x8 (in_buf_uv1.data[index10.w]);
    out_uv10.zw = (x10_fract.w == 0u) ? value.xy : value.zw;
    value = (index11.w == index10.w) ? value : unpackUnorm4x8 (in_buf_uv1.data[index11.w]);
    out_uv11.zw = (x11_fract.w == 0u) ? value.xy : value.zw;

    out_data1 = out_uv00 * weight00.zzww + out_uv01 * weight01.zzww +
                out_uv10 * weight10.zzww + out_uv11 * weight11.zzww;
}

