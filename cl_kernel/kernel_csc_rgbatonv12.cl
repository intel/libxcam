/*
 * function: kernel_csc_rgbatonv12
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * vertical_offset, vertical offset from y to uv
 */

__kernel void kernel_csc_rgbatonv12 (__read_only image2d_t input, __write_only image2d_t output, uint vertical_offset, __global float *matrix)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    float4 pixel_in[8];
    pixel_in[0] = read_imagef(input, sampler, (int2)(4 * x, 2 * y));
    pixel_in[1] = read_imagef(input, sampler, (int2)(4 * x + 1, 2 * y));
    pixel_in[2] = read_imagef(input, sampler, (int2)(4 * x + 2, 2 * y));
    pixel_in[3] = read_imagef(input, sampler, (int2)(4 * x + 3, 2 * y));

    pixel_in[4] = read_imagef(input, sampler, (int2)(4 * x, 2 * y + 1));
    pixel_in[5] = read_imagef(input, sampler, (int2)(4 * x + 1, 2 * y + 1));
    pixel_in[6] = read_imagef(input, sampler, (int2)(4 * x + 2, 2 * y + 1));
    pixel_in[7] = read_imagef(input, sampler, (int2)(4 * x + 3, 2 * y + 1));


    float4 pixel_out[8], pixel_out_u[2], pixel_out_v[2];
    pixel_out[0].x = matrix[0] * pixel_in[0].x + matrix[1] * pixel_in[0].y + matrix[2] * pixel_in[0].z;
    pixel_out[0].y = 0.0;
    pixel_out[0].z = 0.0;
    pixel_out[0].w = 1.0;
    pixel_out[1].x = matrix[0] * pixel_in[1].x + matrix[1] * pixel_in[1].y + matrix[2] * pixel_in[1].z;
    pixel_out[1].y = 0.0;
    pixel_out[1].z = 0.0;
    pixel_out[1].w = 1.0;
    pixel_out[2].x = matrix[0] * pixel_in[2].x + matrix[1] * pixel_in[2].y + matrix[2] * pixel_in[2].z;
    pixel_out[2].y = 0.0;
    pixel_out[2].z = 0.0;
    pixel_out[2].w = 1.0;
    pixel_out[3].x = matrix[0] * pixel_in[3].x + matrix[1] * pixel_in[3].y + matrix[2] * pixel_in[3].z;
    pixel_out[3].y = 0.0;
    pixel_out[3].z = 0.0;
    pixel_out[3].w = 1.0;
    pixel_out[4].x = matrix[0] * pixel_in[4].x + matrix[1] * pixel_in[4].y + matrix[2] * pixel_in[4].z;
    pixel_out[4].y = 0.0;
    pixel_out[4].z = 0.0;
    pixel_out[4].w = 1.0;
    pixel_out[5].x = matrix[0] * pixel_in[5].x + matrix[1] * pixel_in[5].y + matrix[2] * pixel_in[5].z;
    pixel_out[5].y = 0.0;
    pixel_out[5].z = 0.0;
    pixel_out[5].w = 1.0;
    pixel_out[6].x = matrix[0] * pixel_in[6].x + matrix[1] * pixel_in[6].y + matrix[2] * pixel_in[6].z;
    pixel_out[6].y = 0.0;
    pixel_out[6].z = 0.0;
    pixel_out[6].w = 1.0;
    pixel_out[7].x = matrix[0] * pixel_in[7].x + matrix[1] * pixel_in[7].y + matrix[2] * pixel_in[7].z;
    pixel_out[7].y = 0.0;
    pixel_out[7].z = 0.0;
    pixel_out[7].w = 1.0;

    pixel_out_u[0].x = matrix[3] * pixel_in[0].x + matrix[4] * pixel_in[0].y + matrix[5] * pixel_in[0].z + 0.5;
    pixel_out_u[0].y = 0.0;
    pixel_out_u[0].z = 0.0;
    pixel_out_u[0].w = 1.0;
    pixel_out_v[0].x = matrix[6] * pixel_in[0].x + matrix[7] * pixel_in[0].y + matrix[8] * pixel_in[0].z + 0.5;
    pixel_out_v[0].y = 0.0;
    pixel_out_v[0].z = 0.0;
    pixel_out_v[0].w = 1.0;

    pixel_out_u[1].x = matrix[3] * pixel_in[4].x + matrix[4] * pixel_in[4].y + matrix[5] * pixel_in[4].z + 0.5;
    pixel_out_u[1].y = 0.0;
    pixel_out_u[1].z = 0.0;
    pixel_out_u[1].w = 1.0;
    pixel_out_v[1].x = matrix[6] * pixel_in[4].x + matrix[7] * pixel_in[4].y + matrix[8] * pixel_in[4].z + 0.5;
    pixel_out_v[1].y = 0.0;
    pixel_out_v[1].z = 0.0;
    pixel_out_v[1].w = 1.0;

    write_imagef(output, (int2)(4 * x, 2 * y), pixel_out[0]);
    write_imagef(output, (int2)(4 * x + 1, 2 * y), pixel_out[1]);
    write_imagef(output, (int2)(4 * x + 2, 2 * y), pixel_out[2]);
    write_imagef(output, (int2)(4 * x + 3, 2 * y), pixel_out[3]);

    write_imagef(output, (int2)(4 * x, 2 * y + 1), pixel_out[4]);
    write_imagef(output, (int2)(4 * x + 1, 2 * y + 1), pixel_out[5]);
    write_imagef(output, (int2)(4 * x + 2, 2 * y + 1), pixel_out[6]);
    write_imagef(output, (int2)(4 * x + 3, 2 * y + 1), pixel_out[7]);

    write_imagef(output, (int2)(4 * x, y + vertical_offset), pixel_out_u[0]);
    write_imagef(output, (int2)(4 * x + 1, y + vertical_offset), pixel_out_v[0]);
    write_imagef(output, (int2)(4 * x + 2, y + vertical_offset), pixel_out_u[1]);
    write_imagef(output, (int2)(4 * x + 3, y + vertical_offset), pixel_out_v[1]);
}

