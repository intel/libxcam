#version 310 es

layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) readonly buffer InBuf {
    uint data[];
} in_buf;

layout (binding = 1, rgba8) writeonly uniform highp image2D out_texture;

uniform uint in_img_width;

void main ()
{
    uvec2 g_id = gl_GlobalInvocationID.xy;

    vec4 tex = unpackUnorm4x8 (in_buf.data[g_id.y * in_img_width + g_id.x]);

    imageStore (out_texture, ivec2(4u * g_id.x, g_id.y), tex.rrrr);
    imageStore (out_texture, ivec2(4u * g_id.x + 1u, g_id.y), tex.gggg);
    imageStore (out_texture, ivec2(4u * g_id.x + 2u, g_id.y), tex.bbbb);
    imageStore (out_texture, ivec2(4u * g_id.x + 3u, g_id.y), tex.aaaa);
}

