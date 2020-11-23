#version 310 es

layout (local_size_x = 8, local_size_y = 8) in;

layout (binding = 0) readonly buffer In0BufY {
    uvec4 data[];
} in0_buf_y;

layout (binding = 1) readonly buffer In0BufU {
    uvec2 data[];
} in0_buf_u;

layout (binding = 2) readonly buffer In0BufV {
    uvec2 data[];
} in0_buf_v;

layout (binding = 3) readonly buffer In1BufY {
    uvec4 data[];
} in1_buf_y;

layout (binding = 4) readonly buffer In1BufU {
    uvec2 data[];
} in1_buf_u;

layout (binding = 5) readonly buffer In1BufV {
    uvec2 data[];
} in1_buf_v;

layout (binding = 6) writeonly buffer OutBufY {
    uvec4 data[];
} out_buf_y;

layout (binding = 7) writeonly buffer OutBufU {
    uvec2 data[];
} out_buf_u;

layout (binding = 8) writeonly buffer OutBufV {
    uvec2 data[];
} out_buf_v;

layout (binding = 9) readonly buffer MaskBuf {
    uvec4 data[];
} mask_buf;

uniform uint in0_img_width;
uniform uint in1_img_width;
uniform uint out_img_width;

uniform uint blend_width;
uniform uint in0_offset_x;
uniform uint in1_offset_x;
uniform uint out_offset_x;

void main ()
{
    uvec2 g_id = gl_GlobalInvocationID.xy;
    g_id.x = clamp (g_id.x, 0u, blend_width - 1u);

    uvec4 mask = mask_buf.data[g_id.x];
    vec4 mask0 = unpackUnorm4x8 (mask.x);
    vec4 mask1 = unpackUnorm4x8 (mask.y);
    vec4 mask2 = unpackUnorm4x8 (mask.z);
    vec4 mask3 = unpackUnorm4x8 (mask.w);

    uvec3 img_width = uvec3 (in0_img_width, in1_img_width, out_img_width);
    uvec3 offset_x = uvec3 (in0_offset_x, in1_offset_x, out_offset_x);
    uvec3 y_idx = offset_x + g_id.y * 2u * img_width + g_id.x;

    uvec4 in0_y = in0_buf_y.data[y_idx.x];
    vec4 in0_y0 = unpackUnorm4x8 (in0_y.x);
    vec4 in0_y1 = unpackUnorm4x8 (in0_y.y);
    vec4 in0_y2 = unpackUnorm4x8 (in0_y.z);
    vec4 in0_y3 = unpackUnorm4x8 (in0_y.w);

    uvec4 in1_y = in1_buf_y.data[y_idx.y];
    vec4 in1_y0 = unpackUnorm4x8 (in1_y.x);
    vec4 in1_y1 = unpackUnorm4x8 (in1_y.y);
    vec4 in1_y2 = unpackUnorm4x8 (in1_y.z);
    vec4 in1_y3 = unpackUnorm4x8 (in1_y.w);

    vec4 out_y0 = (in0_y0 - in1_y0) * mask0 + in1_y0;
    vec4 out_y1 = (in0_y1 - in1_y1) * mask1 + in1_y1;
    vec4 out_y2 = (in0_y2 - in1_y2) * mask2 + in1_y2;
    vec4 out_y3 = (in0_y3 - in1_y3) * mask3 + in1_y3;
    out_y0 = clamp (out_y0, 0.0f, 1.0f);
    out_y1 = clamp (out_y1, 0.0f, 1.0f);
    out_y2 = clamp (out_y2, 0.0f, 1.0f);
    out_y3 = clamp (out_y3, 0.0f, 1.0f);
    out_buf_y.data[y_idx.z] =
        uvec4 (packUnorm4x8 (out_y0), packUnorm4x8 (out_y1), packUnorm4x8 (out_y2), packUnorm4x8 (out_y3));

    y_idx += img_width;
    in0_y = in0_buf_y.data[y_idx.x];
    in0_y0 = unpackUnorm4x8 (in0_y.x);
    in0_y1 = unpackUnorm4x8 (in0_y.y);
    in0_y2 = unpackUnorm4x8 (in0_y.z);
    in0_y3 = unpackUnorm4x8 (in0_y.w);

    in1_y = in1_buf_y.data[y_idx.y];
    in1_y0 = unpackUnorm4x8 (in1_y.x);
    in1_y1 = unpackUnorm4x8 (in1_y.y);
    in1_y2 = unpackUnorm4x8 (in1_y.z);
    in1_y3 = unpackUnorm4x8 (in1_y.w);

    out_y0 = (in0_y0 - in1_y0) * mask0 + in1_y0;
    out_y1 = (in0_y1 - in1_y1) * mask1 + in1_y1;
    out_y2 = (in0_y2 - in1_y2) * mask2 + in1_y2;
    out_y3 = (in0_y3 - in1_y3) * mask3 + in1_y3;
    out_y0 = clamp (out_y0, 0.0f, 1.0f);
    out_y1 = clamp (out_y1, 0.0f, 1.0f);
    out_y2 = clamp (out_y2, 0.0f, 1.0f);
    out_y3 = clamp (out_y3, 0.0f, 1.0f);
    out_buf_y.data[y_idx.z] =
        uvec4 (packUnorm4x8 (out_y0), packUnorm4x8 (out_y1), packUnorm4x8 (out_y2), packUnorm4x8 (out_y3));

    vec4 mask_uv0 = vec4 (mask0.xz, mask1.xz);
    vec4 mask_uv1 = vec4 (mask2.xz, mask2.xz);
    uvec3 uv_idx = offset_x + g_id.y * img_width + g_id.x;

    uvec2 in0_u = in0_buf_u.data[uv_idx.x];
    vec4 in0_u0 = unpackUnorm4x8 (in0_u.x);
    vec4 in0_u1 = unpackUnorm4x8 (in0_u.y);

    uvec2 in1_u = in1_buf_u.data[uv_idx.x];
    vec4 in1_u0 = unpackUnorm4x8 (in1_u.x);
    vec4 in1_u1 = unpackUnorm4x8 (in1_u.y);

    vec4 out_u0 = (in0_u0 - in1_u0) * mask_uv0 + in1_u0;
    vec4 out_u1 = (in0_u1 - in1_u1) * mask_uv1 + in1_u1;
    out_u0 = clamp (out_u0, 0.0f, 1.0f);
    out_u1 = clamp (out_u1, 0.0f, 1.0f);
    out_buf_u.data[uv_idx.z] = uvec2 (packUnorm4x8 (out_u0), packUnorm4x8 (out_u1));

    uvec2 in0_v = in0_buf_v.data[uv_idx.x];
    vec4 in0_v0 = unpackUnorm4x8 (in0_v.x);
    vec4 in0_v1 = unpackUnorm4x8 (in0_v.y);

    uvec2 in1_v = in1_buf_v.data[uv_idx.x];
    vec4 in1_v0 = unpackUnorm4x8 (in1_v.x);
    vec4 in1_v1 = unpackUnorm4x8 (in1_v.y);

    vec4 out_v0 = (in0_v0 - in1_v0) * mask_uv0 + in1_v0;
    vec4 out_v1 = (in0_v1 - in1_v1) * mask_uv1 + in1_v1;
    out_v0 = clamp (out_v0, 0.0f, 1.0f);
    out_v1 = clamp (out_v1, 0.0f, 1.0f);
    out_buf_v.data[uv_idx.z] = uvec2 (packUnorm4x8 (out_v0), packUnorm4x8 (out_v1));
}
