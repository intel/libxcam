/**
* \brief Image warping kernel function.
* \param[in] input Input image object.
* \param[out] output scaled output image object.
* \param[in] warp_config: image warping parameters
*/

// 8 bytes for each pixel
#define PIXEL_X_STEP   8

typedef struct {
    int frame_id;
    int valid;
    int width;
    int height;
    float trim_ratio;
    float proj_mat[9];
} CLWarpConfig;

#if WRITE_UINT
__kernel void kernel_image_warp (__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 CLWarpConfig warp_config)
{
    // dest coordinate
    int d_x = get_global_id(0);
    int d_y = get_global_id(1);

    size_t width = get_global_size(0);
    size_t height = get_global_size(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // source coordinate
    float s_x = 0.0f;
    float s_y = 0.0f;
    float w = 0.0f;
    float warp_x = 0.0f;
    float warp_y = 0.0f;

    // trim coordinate
    float t_x = 0.0f;
    float t_y = 0.0f;

    float8 pixel = 0.0f;
    float* output_pixel = (float*)(&pixel);
    int i = 0;

#if WARP_Y
    t_y = (1.0f - 2.0f * warp_config.trim_ratio) * (float)d_y +
          warp_config.trim_ratio * (float)height;

    for (i = 0; i < PIXEL_X_STEP; i++) {
        t_x = (1.0f - 2.0f * warp_config.trim_ratio) * (float)(PIXEL_X_STEP * d_x + i)
              + warp_config.trim_ratio * (float)(PIXEL_X_STEP * width);

        s_x = warp_config.proj_mat[0] * t_x +
              warp_config.proj_mat[1] * t_y +
              warp_config.proj_mat[2];
        s_y = warp_config.proj_mat[3] * t_x +
              warp_config.proj_mat[4] * t_y +
              warp_config.proj_mat[5];
        w = warp_config.proj_mat[6] * t_x +
            warp_config.proj_mat[7] * t_y +
            warp_config.proj_mat[8];
        w = w != 0.0f ? 1.0f / w : 0.0f;

        warp_x = (s_x * w) / (float)(PIXEL_X_STEP * width);
        warp_y = (s_y * w) / (float)height;

        output_pixel[i] = read_imagef(input, sampler, (float2)(warp_x, warp_y)).x;
    }
    write_imageui(output, (int2)(d_x, d_y), convert_uint4(as_ushort4(convert_uchar8(pixel * 255.0f))));
#endif

#if WARP_UV
    t_y = (1.0f - 2.0f * warp_config.trim_ratio) * (float)d_y + warp_config.trim_ratio * (float)height;

    for (i = 0; i < (PIXEL_X_STEP >> 1); i++) {
        t_x = (1.0f - 2.0f * warp_config.trim_ratio) * (float)(PIXEL_X_STEP * d_x + 2 * i)
              + warp_config.trim_ratio * (float)(PIXEL_X_STEP * width);

        s_x = warp_config.proj_mat[0] * t_x +
              warp_config.proj_mat[1] * t_y +
              warp_config.proj_mat[2];
        s_y = warp_config.proj_mat[3] * t_x +
              warp_config.proj_mat[4] * t_y +
              warp_config.proj_mat[5];
        w = warp_config.proj_mat[6] * t_x +
            warp_config.proj_mat[7] * t_y +
            warp_config.proj_mat[8];
        w = w != 0.0f ? 1.0f / w : 0.0f;
        warp_x = (s_x * w) / (float)(PIXEL_X_STEP * width);

        warp_y = (s_y * w) / (float)height;

        float2 temp = read_imagef(input, sampler, (float2)(warp_x, warp_y)).xy;
        output_pixel[2 * i] = temp.x;
        output_pixel[2 * i + 1] = temp.y;
    }
    write_imageui(output, (int2)(d_x, d_y), convert_uint4(as_ushort4(convert_uchar8(pixel * 255.0f))));
#endif

}

#else
__kernel void kernel_image_warp (__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 CLWarpConfig warp_config)
{
    // dest coordinate
    int d_x = get_global_id(0);
    int d_y = get_global_id(1);

    size_t width = get_global_size(0);
    size_t height = get_global_size(1);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // trim coordinate
    float t_x = (1.0f - 2.0f * warp_config.trim_ratio) * (float)d_x +
                warp_config.trim_ratio * (float)width;
    float t_y = (1.0f - 2.0f * warp_config.trim_ratio) * (float)d_y +
                warp_config.trim_ratio * (float)height;

    // source coordinate
    float s_x = warp_config.proj_mat[0] * t_x +
                warp_config.proj_mat[1] * t_y +
                warp_config.proj_mat[2];
    float s_y = warp_config.proj_mat[3] * t_x +
                warp_config.proj_mat[4] * t_y +
                warp_config.proj_mat[5];
    float w = warp_config.proj_mat[6] * t_x +
              warp_config.proj_mat[7] * t_y +
              warp_config.proj_mat[8];
    w = w != 0.0f ? 1.0f / w : 0.0f;

    float warp_x = (s_x * w) / (float)width;
    float warp_y = (s_y * w) / (float)height;

    float4 pixel = read_imagef(input, sampler, (float2)(warp_x, warp_y));

    write_imagef(output, (int2)(d_x, d_y), pixel);
}
#endif


