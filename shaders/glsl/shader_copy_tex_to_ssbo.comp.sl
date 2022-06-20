#version 310 es

layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) uniform sampler2D in_texture;
layout (binding = 1) writeonly buffer OutBuf {
    uint data[];
} out_buf;

uniform uint out_img_width;

void main ()
{
    uvec2 g_id = gl_GlobalInvocationID.xy;

    vec4 out1 = texelFetch (in_texture, ivec2(4u * g_id.x, g_id.y), 0);
    vec4 out2 = texelFetch (in_texture, ivec2(4u * g_id.x + 1u, g_id.y), 0);
    vec4 out3 = texelFetch (in_texture, ivec2(4u * g_id.x + 2u, g_id.y), 0);
    vec4 out4 = texelFetch (in_texture, ivec2(4u * g_id.x + 3u, g_id.y), 0);

    vec4 out_pixels = vec4 (out1.r, out2.r, out3.r, out4.r);

    out_buf.data[g_id.y * out_img_width + g_id.x] = packUnorm4x8 (out_pixels);
}

