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

layout (binding = 3) readonly buffer GaussScaleBufY {
    uvec2 data[];
} gaussscale_buf_y;

layout (binding = 4) readonly buffer GaussScaleBufU {
    uint data[];
} gaussscale_buf_u;

layout (binding = 5) readonly buffer GaussScaleBufV {
    uint data[];
} gaussscale_buf_v;

layout (binding = 6) writeonly buffer OutBufY {
    uvec4 data[];
} out_buf_y;

layout (binding = 7) writeonly buffer OutBufU {
    uvec2 data[];
} out_buf_u;

layout (binding = 8) writeonly buffer OutBufV {
    uvec2 data[];
} out_buf_v;

uniform uint in_img_width;
uniform uint in_img_height;
uniform uint in_offset_x;

uniform uint gaussscale_img_width;
uniform uint gaussscale_img_height;

uniform uint merge_width;

// normalization of half gray level
const float norm_half_gl = 128.0f / 255.0f;

void lap_trans_y (uvec2 y_id, uvec2 gs_id);
void lap_trans_uv (uvec2 uv_id, uvec2 gs_id);

void main ()
{
    uvec2 g_id = gl_GlobalInvocationID.xy;

    uvec2 y_id = uvec2 (g_id.x, g_id.y * 4u);
    y_id.x = clamp (y_id.x, 0u, merge_width - 1u);

    uvec2 gs_id = uvec2 (g_id.x, g_id.y * 2u);
    gs_id.x = clamp (gs_id.x, 0u, gaussscale_img_width - 1u);
    lap_trans_y (y_id, gs_id);

    y_id.y += 2u;
    gs_id.y += 1u;
    lap_trans_y (y_id, gs_id);

    uvec2 uv_id = uvec2 (y_id.x, g_id.y * 2u);
    gs_id.y = g_id.y;
    lap_trans_uv (uv_id, gs_id);
}

void lap_trans_y (uvec2 y_id, uvec2 gs_id)
{
    y_id.y = clamp (y_id.y, 0u, in_img_height - 1u);
    gs_id.y = clamp (gs_id.y, 0u, gaussscale_img_height - 1u);

    uint y_idx = y_id.y * in_img_width + in_offset_x + y_id.x;
    uvec4 in_pack = in_buf_y.data[y_idx];
    vec4 in0 = unpackUnorm4x8 (in_pack.x);
    vec4 in1 = unpackUnorm4x8 (in_pack.y);
    vec4 in2 = unpackUnorm4x8 (in_pack.z);
    vec4 in3 = unpackUnorm4x8 (in_pack.w);

    uint gs_idx = gs_id.y * gaussscale_img_width + gs_id.x;
    uvec2 gs = gaussscale_buf_y.data[gs_idx];
    vec4 gs0 = unpackUnorm4x8 (gs.x);
    vec4 gs1 = unpackUnorm4x8 (gs.y);
    gs = gaussscale_buf_y.data[gs_idx + 1u];
    vec4 gs2 = unpackUnorm4x8 (gs.x);
    gs2 = (gs_id.x == gaussscale_img_width - 1u) ? gs1.wwww : gs2;

    vec4 inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter00 = vec4 (gs0.x, inter.x, gs0.y, inter.y);
    vec4 inter01 = vec4 (gs0.z, inter.z, gs0.w, inter.w);
    inter = (gs1 + vec4 (gs1.yzw, gs2.x)) * 0.5f;
    vec4 inter02 = vec4 (gs1.x, inter.x, gs1.y, inter.y);
    vec4 inter03 = vec4 (gs1.z, inter.z, gs1.w, inter.w);

    vec4 lap0 = (in0 - inter00) * 0.5f + norm_half_gl;
    vec4 lap1 = (in1 - inter01) * 0.5f + norm_half_gl;
    vec4 lap2 = (in2 - inter02) * 0.5f + norm_half_gl;
    vec4 lap3 = (in3 - inter03) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);
    lap2 = clamp (lap2, 0.0f, 1.0f);
    lap3 = clamp (lap3, 0.0f, 1.0f);

    uint out_idx = y_id.y * merge_width + y_id.x;
    out_buf_y.data[out_idx] =
        uvec4 (packUnorm4x8 (lap0), packUnorm4x8 (lap1), packUnorm4x8 (lap2), packUnorm4x8 (lap3));

    y_idx = (y_id.y >= in_img_height - 1u) ? y_idx : y_idx + in_img_width;
    in_pack = in_buf_y.data[y_idx];
    in0 = unpackUnorm4x8 (in_pack.x);
    in1 = unpackUnorm4x8 (in_pack.y);
    in2 = unpackUnorm4x8 (in_pack.z);
    in3 = unpackUnorm4x8 (in_pack.w);

    gs_idx = (gs_id.y >= gaussscale_img_height - 1u) ? gs_idx : gs_idx + gaussscale_img_width;
    gs = gaussscale_buf_y.data[gs_idx];
    gs0 = unpackUnorm4x8 (gs.x);
    gs1 = unpackUnorm4x8 (gs.y);
    gs = gaussscale_buf_y.data[gs_idx + 1u];
    gs2 = unpackUnorm4x8 (gs.x);
    gs2 = (gs_id.x == gaussscale_img_width - 1u) ? gs1.wwww : gs2;

    inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter10 = (inter00 + vec4 (gs0.x, inter.x, gs0.y, inter.y)) * 0.5f;
    vec4 inter11 = (inter01 + vec4 (gs0.z, inter.z, gs0.w, inter.w)) * 0.5f;
    inter = (gs1 + vec4 (gs1.yzw, gs2.x)) * 0.5f;
    vec4 inter12 = (inter02 + vec4 (gs1.x, inter.x, gs1.y, inter.y)) * 0.5f;
    vec4 inter13 = (inter03 + vec4 (gs1.z, inter.z, gs1.w, inter.w)) * 0.5f;

    lap0 = (in0 - inter10) * 0.5f + norm_half_gl;
    lap1 = (in1 - inter11) * 0.5f + norm_half_gl;
    lap2 = (in2 - inter12) * 0.5f + norm_half_gl;
    lap3 = (in3 - inter13) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);
    lap2 = clamp (lap2, 0.0f, 1.0f);
    lap3 = clamp (lap3, 0.0f, 1.0f);

    out_idx += merge_width;
    out_buf_y.data[out_idx] =
        uvec4 (packUnorm4x8 (lap0), packUnorm4x8 (lap1), packUnorm4x8 (lap2), packUnorm4x8 (lap3));
}

void lap_trans_uv (uvec2 uv_id, uvec2 gs_id)
{
    uv_id.y = clamp (uv_id.y, 0u, (in_img_height >> 1u) - 1u);
    gs_id.y = clamp (gs_id.y, 0u, (gaussscale_img_height >> 1u) - 1u);

    uint gs_idx = gs_id.y * gaussscale_img_width + gs_id.x;
    vec4 gs0 = unpackUnorm4x8 (gaussscale_buf_u.data[gs_idx]);
    vec4 gs1 = unpackUnorm4x8 (gaussscale_buf_u.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.wwww : gs1;

    vec4 inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter_u00 = vec4 (gs0.x, inter.x, gs0.y, inter.y);
    vec4 inter_u01 = vec4 (gs0.z, inter.z, gs0.w, inter.w);

    uint uv_idx = uv_id.y * in_img_width + in_offset_x + uv_id.x;
    uvec2 in_pack = in_buf_u.data[uv_idx];
    vec4 in0 = unpackUnorm4x8 (in_pack.x);
    vec4 in1 = unpackUnorm4x8 (in_pack.y);

    vec4 lap0 = (in0 - inter_u00) * 0.5f + norm_half_gl;
    vec4 lap1 = (in1 - inter_u01) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    uint out_idx = uv_id.y * merge_width + uv_id.x;
    out_buf_u.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));

    gs0 = unpackUnorm4x8 (gaussscale_buf_v.data[gs_idx]);
    gs1 = unpackUnorm4x8 (gaussscale_buf_v.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.wwww : gs1;

    inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter_v00 = vec4 (gs0.x, inter.x, gs0.y, inter.y);
    vec4 inter_v01 = vec4 (gs0.z, inter.z, gs0.w, inter.w);

    in_pack = in_buf_v.data[uv_idx];
    in0 = unpackUnorm4x8 (in_pack.x);
    in1 = unpackUnorm4x8 (in_pack.y);

    lap0 = (in0 - inter_v00) * 0.5f + norm_half_gl;
    lap1 = (in1 - inter_v01) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    out_buf_v.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));

    gs_idx = (gs_id.y >= (gaussscale_img_height >> 1u) - 1u) ? gs_idx : gs_idx + gaussscale_img_width;
    gs0 = unpackUnorm4x8 (gaussscale_buf_u.data[gs_idx]);
    gs1 = unpackUnorm4x8 (gaussscale_buf_u.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.wwww : gs1;

    inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter_u10 = (inter_u00 + vec4 (gs0.x, inter.x, gs0.y, inter.y)) * 0.5f;
    vec4 inter_u11 = (inter_u01 + vec4 (gs0.z, inter.z, gs0.w, inter.w)) * 0.5f;

    uv_idx = (uv_id.y >= (in_img_height >> 1u) - 1u) ? uv_idx : uv_idx + in_img_width;
    in_pack = in_buf_u.data[uv_idx];
    in0 = unpackUnorm4x8 (in_pack.x);
    in1 = unpackUnorm4x8 (in_pack.y);

    lap0 = (in0 - inter_u10) * 0.5f + norm_half_gl;
    lap1 = (in1 - inter_u11) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    out_idx += merge_width;
    out_buf_u.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));

    gs0 = unpackUnorm4x8 (gaussscale_buf_v.data[gs_idx]);
    gs1 = unpackUnorm4x8 (gaussscale_buf_v.data[gs_idx + 1u]);
    gs1 = (gs_id.x == gaussscale_img_width - 1u) ? gs0.wwww : gs1;

    inter = (gs0 + vec4 (gs0.yzw, gs1.x)) * 0.5f;
    vec4 inter_v10 = (inter_v00 + vec4 (gs0.x, inter.x, gs0.y, inter.y)) * 0.5f;
    vec4 inter_v11 = (inter_v01 + vec4 (gs0.z, inter.z, gs0.w, inter.w)) * 0.5f;

    in_pack = in_buf_v.data[uv_idx];
    in0 = unpackUnorm4x8 (in_pack.x);
    in1 = unpackUnorm4x8 (in_pack.y);

    lap0 = (in0 - inter_v10) * 0.5f + norm_half_gl;
    lap1 = (in1 - inter_v11) * 0.5f + norm_half_gl;
    lap0 = clamp (lap0, 0.0f, 1.0f);
    lap1 = clamp (lap1, 0.0f, 1.0f);

    out_buf_v.data[out_idx] = uvec2 (packUnorm4x8 (lap0), packUnorm4x8 (lap1));
}
