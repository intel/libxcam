#version 310 es

layout (local_size_x = 8, local_size_y = 8) in;

layout (binding = 0) readonly buffer In0BufY {
    uvec2 data[];
} in0_buf_y;

layout (binding = 1) readonly buffer In0BufUV {
    uvec2 data[];
} in0_buf_uv;

layout (binding = 2) readonly buffer In1BufY {
    uvec2 data[];
} in1_buf_y;

layout (binding = 3) readonly buffer In1BufUV {
    uvec2 data[];
} in1_buf_uv;

layout (binding = 4) writeonly buffer OutBufY {
    uvec2 data[];
} out_buf_y;

layout (binding = 5) writeonly buffer OutBufUV {
    uvec2 data[];
} out_buf_uv;

layout (binding = 6) readonly buffer MaskBuf {
    uvec2 data[];
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

    uvec2 mask = mask_buf.data[g_id.x];
    vec4 mask0 = unpackUnorm4x8 (mask.x);
    vec4 mask1 = unpackUnorm4x8 (mask.y);

    uvec3 img_width = uvec3 (in0_img_width, in1_img_width, out_img_width);
    uvec3 offset_x = uvec3 (in0_offset_x, in1_offset_x, out_offset_x);
    uvec3 y_idx = g_id.y * 2u * img_width + offset_x + g_id.x;

    uvec2 in0_y = in0_buf_y.data[y_idx.x];
    vec4 in0_y0 = unpackUnorm4x8 (in0_y.x);
    vec4 in0_y1 = unpackUnorm4x8 (in0_y.y);

    uvec2 in1_y = in1_buf_y.data[y_idx.y];
    vec4 in1_y0 = unpackUnorm4x8 (in1_y.x);
    vec4 in1_y1 = unpackUnorm4x8 (in1_y.y);

    vec4 out_y0 = (in0_y0 - in1_y0) * mask0 + in1_y0;
    vec4 out_y1 = (in0_y1 - in1_y1) * mask1 + in1_y1;
    out_y0 = clamp (out_y0, 0.0f, 1.0f);
    out_y1 = clamp (out_y1, 0.0f, 1.0f);
    out_buf_y.data[y_idx.z] = uvec2 (packUnorm4x8 (out_y0), packUnorm4x8 (out_y1));

    y_idx += img_width;
    in0_y = in0_buf_y.data[y_idx.x];
    in0_y0 = unpackUnorm4x8 (in0_y.x);
    in0_y1 = unpackUnorm4x8 (in0_y.y);

    in1_y = in1_buf_y.data[y_idx.y];
    in1_y0 = unpackUnorm4x8 (in1_y.x);
    in1_y1 = unpackUnorm4x8 (in1_y.y);

    out_y0 = (in0_y0 - in1_y0) * mask0 + in1_y0;
    out_y1 = (in0_y1 - in1_y1) * mask1 + in1_y1;
    out_y0 = clamp (out_y0, 0.0f, 1.0f);
    out_y1 = clamp (out_y1, 0.0f, 1.0f);
    out_buf_y.data[y_idx.z] = uvec2 (packUnorm4x8 (out_y0), packUnorm4x8 (out_y1));

    uvec3 uv_idx = g_id.y * img_width + offset_x + g_id.x;
    uvec2 in0_uv = in0_buf_uv.data[uv_idx.x];
    vec4 in0_uv0 = unpackUnorm4x8 (in0_uv.x);
    vec4 in0_uv1 = unpackUnorm4x8 (in0_uv.y);

    uvec2 in1_uv = in1_buf_uv.data[uv_idx.y];
    vec4 in1_uv0 = unpackUnorm4x8 (in1_uv.x);
    vec4 in1_uv1 = unpackUnorm4x8 (in1_uv.y);

    mask0.yw = mask0.xz;
    mask1.yw = mask1.xz;
    vec4 out_uv0 = (in0_uv0 - in1_uv0) * mask0 + in1_uv0;
    vec4 out_uv1 = (in0_uv1 - in1_uv1) * mask1 + in1_uv1;
    out_uv0 = clamp (out_uv0, 0.0f, 1.0f);
    out_uv1 = clamp (out_uv1, 0.0f, 1.0f);
    out_buf_uv.data[uv_idx.z] = uvec2 (packUnorm4x8 (out_uv0), packUnorm4x8 (out_uv1));
}
