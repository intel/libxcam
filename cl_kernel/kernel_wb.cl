/*
 * function: kernel_wb
 *     black level correction for sensor data input
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * wb_config: white balance configuration
 */

typedef struct
{
    float r_gain;
    float gr_gain;
    float gb_gain;
    float b_gain;
} CLWBConfig;

__kernel void kernel_wb (__read_only image2d_t input,
                         __write_only image2d_t output,
                         CLWBConfig wb_config)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 Gr_in[2], R_in[2], B_in[2], Gb_in[2];
    float4 Gr_out[2], R_out[2], B_out[2], Gb_out[2];
    Gr_in[0] = read_imagef(input, sampler, (int2)(4 * x, 2 * y));
    R_in[0] = read_imagef(input, sampler, (int2)(4 * x + 1, 2 * y));
    B_in[0] = read_imagef(input, sampler, (int2)(4 * x, 2 * y + 1));
    Gb_in[0] = read_imagef(input, sampler, (int2)(4 * x + 1, 2 * y + 1));
    Gr_out[0].x = Gr_in[0].x * wb_config.gr_gain;
    Gr_out[0].y = 0.0;
    Gr_out[0].z = 0.0;
    Gr_out[0].w = 1.0;
    R_out[0].x = R_in[0].x * wb_config.r_gain;
    R_out[0].y = 0.0;
    R_out[0].z = 0.0;
    R_out[0].w = 1.0;
    B_out[0].x = B_in[0].x * wb_config.b_gain;
    B_out[0].y = 0.0;
    B_out[0].z = 0.0;
    B_out[0].w = 1.0;
    Gb_out[0].x = Gb_in[0].x * wb_config.gb_gain;
    Gb_out[0].y = 0.0;
    Gb_out[0].z = 0.0;
    Gb_out[0].w = 1.0;
    write_imagef(output, (int2)(4 * x, 2 * y), Gr_out[0]);
    write_imagef(output, (int2)(4 * x + 1, 2 * y), R_out[0]);
    write_imagef(output, (int2)(4 * x, 2 * y + 1), B_out[0]);
    write_imagef(output, (int2)(4 * x + 1, 2 * y + 1), Gb_out[0]);

    Gr_in[1] = read_imagef(input, sampler, (int2)(4 * x + 2, 2 * y));
    R_in[1] = read_imagef(input, sampler, (int2)(4 * x + 3, 2 * y));
    B_in[1] = read_imagef(input, sampler, (int2)(4 * x + 2, 2 * y + 1));
    Gb_in[1] = read_imagef(input, sampler, (int2)(4 * x + 3, 2 * y + 1));
    Gr_out[1].x = Gr_in[1].x * wb_config.gr_gain;
    Gr_out[1].y = 0.0;
    Gr_out[1].z = 0.0;
    Gr_out[1].w = 1.0;
    R_out[1].x = R_in[1].x * wb_config.r_gain;
    R_out[1].y = 0.0;
    R_out[1].z = 0.0;
    R_out[1].w = 1.0;
    B_out[1].x = B_in[1].x * wb_config.b_gain;
    B_out[1].y = 0.0;
    B_out[1].z = 0.0;
    B_out[1].w = 1.0;
    Gb_out[1].x = Gb_in[1].x * wb_config.gb_gain;
    Gb_out[1].y = 0.0;
    Gb_out[1].z = 0.0;
    Gb_out[1].w = 1.0;
    write_imagef(output, (int2)(4 * x + 2, 2 * y), Gr_out[1]);
    write_imagef(output, (int2)(4 * x + 3, 2 * y), R_out[1]);
    write_imagef(output, (int2)(4 * x + 2, 2 * y + 1), B_out[1]);
    write_imagef(output, (int2)(4 * x + 3, 2 * y + 1), Gb_out[1]);

}
