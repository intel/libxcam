/*
 * function: kernel_gamma
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * table: gamma table.
 */
__kernel void kernel_gamma (__read_only image2d_t input, __write_only image2d_t output, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in[8], pixel_out[8];
    pixel_in[0] = read_imagef(input, sampler,(int2)(4*x,2*y));
    pixel_in[1] = read_imagef(input, sampler,(int2)(4*x+1,2*y));
    pixel_in[2] = read_imagef(input, sampler,(int2)(4*x+2,2*y));
    pixel_in[3] = read_imagef(input, sampler,(int2)(4*x+3,2*y));

    pixel_in[4] = read_imagef(input, sampler,(int2)(4*x,2*y+1));
    pixel_in[5] = read_imagef(input, sampler,(int2)(4*x+1,2*y+1));
    pixel_in[6] = read_imagef(input, sampler,(int2)(4*x+2,2*y+1));
    pixel_in[7] = read_imagef(input, sampler,(int2)(4*x+3,2*y+1));

    pixel_out[0].x = table[convert_int(pixel_in[0].x * 255.0)] / 255.0;
    pixel_out[0].y = table[convert_int(pixel_in[0].y * 255.0)] / 255.0;
    pixel_out[0].z = table[convert_int(pixel_in[0].z * 255.0)] / 255.0;
    pixel_out[0].w = 0.0;

    pixel_out[1].x = table[convert_int(pixel_in[1].x * 255.0)] / 255.0;
    pixel_out[1].y = table[convert_int(pixel_in[1].y * 255.0)] / 255.0;
    pixel_out[1].z = table[convert_int(pixel_in[1].z * 255.0)] / 255.0;
    pixel_out[1].w = 0.0;

    pixel_out[2].x = table[convert_int(pixel_in[2].x * 255.0)] / 255.0;
    pixel_out[2].y = table[convert_int(pixel_in[2].y * 255.0)] / 255.0;
    pixel_out[2].z = table[convert_int(pixel_in[2].z * 255.0)] / 255.0;
    pixel_out[2].w = 0.0;

    pixel_out[3].x = table[convert_int(pixel_in[3].x * 255.0)] / 255.0;
    pixel_out[3].y = table[convert_int(pixel_in[3].y * 255.0)] / 255.0;
    pixel_out[3].z = table[convert_int(pixel_in[3].z * 255.0)] / 255.0;
    pixel_out[3].w = 0.0;

    pixel_out[4].x = table[convert_int(pixel_in[4].x * 255.0)] / 255.0;
    pixel_out[4].y = table[convert_int(pixel_in[4].y * 255.0)] / 255.0;
    pixel_out[4].z = table[convert_int(pixel_in[4].z * 255.0)] / 255.0;
    pixel_out[4].w = 0.0;

    pixel_out[5].x = table[convert_int(pixel_in[5].x * 255.0)] / 255.0;
    pixel_out[5].y = table[convert_int(pixel_in[5].y * 255.0)] / 255.0;
    pixel_out[5].z = table[convert_int(pixel_in[5].z * 255.0)] / 255.0;
    pixel_out[5].w = 0.0;

    pixel_out[6].x = table[convert_int(pixel_in[6].x * 255.0)] / 255.0;
    pixel_out[6].y = table[convert_int(pixel_in[6].y * 255.0)] / 255.0;
    pixel_out[6].z = table[convert_int(pixel_in[6].z * 255.0)] / 255.0;
    pixel_out[6].w = 0.0;

    pixel_out[7].x = table[convert_int(pixel_in[7].x * 255.0)] / 255.0;
    pixel_out[7].y = table[convert_int(pixel_in[7].y * 255.0)] / 255.0;
    pixel_out[7].z = table[convert_int(pixel_in[7].z * 255.0)] / 255.0;
    pixel_out[7].w = 0.0;

    write_imagef(output, (int2)(4*x,2*y), pixel_out[0]);
    write_imagef(output, (int2)(4*x+1,2*y), pixel_out[1]);
    write_imagef(output, (int2)(4*x+2,2*y), pixel_out[2]);
    write_imagef(output, (int2)(4*x+3,2*y), pixel_out[3]);

    write_imagef(output, (int2)(4*x,2*y+1), pixel_out[4]);
    write_imagef(output, (int2)(4*x+1,2*y+1), pixel_out[5]);
    write_imagef(output, (int2)(4*x+2,2*y+1), pixel_out[6]);
    write_imagef(output, (int2)(4*x+3,2*y+1), pixel_out[7]);
}

