#version 310 es

layout (local_size_x = 2, local_size_y = 4) in;

layout (binding = 0) readonly buffer InBufU0 {
    uint data[];
} in_buf_u0;

layout (binding = 1) readonly buffer InBufV0 {
    uint data[];
} in_buf_v0;

layout (binding = 2) readonly buffer CoordX0 {
    vec4 data[];
} coordx0;

layout (binding = 3) readonly buffer CoordY0 {
    vec4 data[];
} coordy0;

layout (binding = 4) readonly buffer InBufU1 {
    uint data[];
} in_buf_u1;

layout (binding = 5) readonly buffer InBufV1 {
    uint data[];
} in_buf_v1;

layout (binding = 6) readonly buffer CoordX1 {
    vec4 data[];
} coordx1;

layout (binding = 7) readonly buffer CoordY1 {
    vec4 data[];
} coordy1;

layout (binding = 8) writeonly buffer OutBufU {
    uint data[];
} out_buf_u;

layout (binding = 9) writeonly buffer OutBufV {
    uint data[];
} out_buf_v;

layout (binding = 10) readonly buffer MaskBuf {
    uint data[];
} mask_buf;

uniform uint in_img_width0;
uniform uint in_img_width1;
uniform uint out_img_width;
uniform uint out_offset_x;

uniform uint coords_width0;
uniform uint coords_width1;

void geomap_uv0 (uint coord_pos, out vec4 out_u, out vec4 out_v);
void geomap_uv1 (uint coord_pos, out vec4 out_u, out vec4 out_v);

void main ()
{
    uvec2 coord_pos = gl_GlobalInvocationID.y * uvec2 (coords_width0, coords_width1) + gl_GlobalInvocationID.x;

    vec4 out_u0, out_v0;
    geomap_uv0 (coord_pos.x, out_u0, out_v0);

    vec4 out_u1, out_v1;
    geomap_uv1 (coord_pos.y, out_u1, out_v1);

    vec4 mask = unpackUnorm4x8 (mask_buf.data[gl_GlobalInvocationID.x]);

    vec4 out_u = (out_u0 - out_u1) * mask + out_u1;
    uint out_pos = gl_GlobalInvocationID.y * out_img_width + out_offset_x + gl_GlobalInvocationID.x;
    out_buf_u.data[out_pos] = packUnorm4x8 (out_u);

    vec4 out_v = (out_v0 - out_v1) * mask + out_v1;
    out_buf_v.data[out_pos] = packUnorm4x8 (out_v);
}

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

void geomap_uv0 (uint coord_pos, out vec4 out_u, out vec4 out_v)
{
    vec4 in_img_x = coordx0.data[coord_pos];
    vec4 in_img_y = coordy0.data[coord_pos];

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

    // pixel U-value
    vec4 out00, out01, out10, out11;
    unpack_unorm (in_buf_u0, 0);
    unpack_unorm (in_buf_u0, 1);
    unpack_unorm (in_buf_u0, 2);
    unpack_unorm (in_buf_u0, 3);
    out_u = out00 * weight00 + out01 * weight01 + out10 * weight10 + out11 * weight11;

    // pixel V-value
    unpack_unorm (in_buf_v0, 0);
    unpack_unorm (in_buf_v0, 1);
    unpack_unorm (in_buf_v0, 2);
    unpack_unorm (in_buf_v0, 3);
    out_v = out00 * weight00 + out01 * weight01 + out10 * weight10 + out11 * weight11;
}

void geomap_uv1 (uint coord_pos, out vec4 out_u, out vec4 out_v)
{
    vec4 in_img_x = coordx1.data[coord_pos];
    vec4 in_img_y = coordy1.data[coord_pos];

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

    // pixel U-value
    vec4 out00, out01, out10, out11;
    unpack_unorm (in_buf_u1, 0);
    unpack_unorm (in_buf_u1, 1);
    unpack_unorm (in_buf_u1, 2);
    unpack_unorm (in_buf_u1, 3);
    out_u = out00 * weight00 + out01 * weight01 + out10 * weight10 + out11 * weight11;

    // pixel V-value
    unpack_unorm (in_buf_v1, 0);
    unpack_unorm (in_buf_v1, 1);
    unpack_unorm (in_buf_v1, 2);
    unpack_unorm (in_buf_v1, 3);
    out_v = out00 * weight00 + out01 * weight01 + out10 * weight10 + out11 * weight11;
}

