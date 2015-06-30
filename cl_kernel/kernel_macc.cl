/*
 * function: kernel_macc
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * table: macc table.
 */
unsigned int get_sector_id (float u, float v)
{
    if ((u >= 0.0) && (v >= 0.0 ))
    {
        if (v / u <= 0.5)
            return 0;
        else if ((v / u > .05) && (v / u <= 1.0))
            return 1;
        else if ((v / u > 1.0) && (v / u <= 2.0))
            return 2;
        else
            return 3;
    }
    else if ((u < 0.0) && (v >= 0.0))
    {
        if (v / u <= -2.0)
            return 4;
        if ((v / u > -2.0) && (v / u <= -1.0))
            return 5;
        if ((v / u > -1.0) && (v / u <= -0.5))
            return 6;
        else
            return 7;
    }
    else if ((u < 0.0) && (v <= 0.0))
    {
        if (v / u <= 0.5)
            return 8;
        else if ((v / u > 0.5) && (v / u <= 1.0))
            return 9;
        else if ((v / u > 1.0) && (v / u <= 2.0))
            return 10;
        else
            return 11;
    }
    else
    {
        if(v / u <= -2.0)
            return 12;
        else if((v / u > -2.0) && (v / u <= -1.0))
            return 13;
        else if((v / u > -1.0) && (v / u <= -0.5))
            return 14;
        else
            return 15;
    }
}
__kernel void kernel_macc (__read_only image2d_t input, __write_only image2d_t output, __global float *table)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 pixel_in[8], pixel_out[8];
    pixel_in[0] = read_imagef(input, sampler, (int2)(4*x,2*y));
    pixel_in[1] = read_imagef(input, sampler, (int2)(4*x+1,2*y));
    pixel_in[2] = read_imagef(input, sampler, (int2)(4*x+2,2*y));
    pixel_in[3] = read_imagef(input, sampler, (int2)(4*x+3,2*y));

    pixel_in[4] = read_imagef(input, sampler, (int2)(4*x,2*y+1));
    pixel_in[5] = read_imagef(input, sampler, (int2)(4*x+1,2*y+1));
    pixel_in[6] = read_imagef(input, sampler, (int2)(4*x+2,2*y+1));
    pixel_in[7] = read_imagef(input, sampler, (int2)(4*x+3,2*y+1));
    float Y[8], ui[8], vi[8], uo[8], vo[8];
    unsigned int table_id[8];
    Y[0] = 0.3 * pixel_in[0].x + 0.59 * pixel_in[0].y + 0.11 * pixel_in[0].z;
    ui[0] = 0.493 * (pixel_in[0].z - Y[0]);
    vi[0] = 0.877 * (pixel_in[0].x - Y[0]);
    table_id[0] = get_sector_id(ui[0], vi[0]);
    uo[0] = ui[0] * table[4 * table_id[0]] + vi[0] * table[4 * table_id[0] + 1];
    vo[0] = ui[0] * table[4 * table_id[0] + 2] + vi[0] * table[4 * table_id[0] + 3];
    pixel_out[0].x = Y[0] + 1.14 * vo[0];
    pixel_out[0].y = Y[0] - 0.39 * uo[0] - 0.58 * vo[0];
    pixel_out[0].z = Y[0] + 2.03 * uo[0];
    pixel_out[0].w = 0.0;
    write_imagef(output, (int2)(4*x,2*y), pixel_out[0]);

    Y[1] = 0.3 * pixel_in[1].x + 0.59 * pixel_in[1].y + 0.11 * pixel_in[1].z;
    ui[1] = 0.493 * (pixel_in[1].z - Y[1]);
    vi[1] = 0.877 * (pixel_in[1].x - Y[1]);
    table_id[1] = get_sector_id(ui[1], vi[1]);
    uo[1] = ui[1] * table[4 * table_id[1]] + vi[1] * table[4 * table_id[1] + 1];
    vo[1] = ui[1] * table[4 * table_id[1] + 2] + vi[1] * table[4 * table_id[1] + 3];
    pixel_out[1].x = Y[1] + 1.14 * vo[1];
    pixel_out[1].y = Y[1] - 0.39 * uo[1] - 0.58 * vo[1];
    pixel_out[1].z = Y[1] + 2.03 * uo[1];
    pixel_out[1].w = 0.0;
    write_imagef(output, (int2)(4*x+1,2*y), pixel_out[1]);

    Y[2] = 0.3 * pixel_in[2].x + 0.59 * pixel_in[2].y + 0.11 * pixel_in[2].z;
    ui[2] = 0.493 * (pixel_in[2].z - Y[2]);
    vi[2] = 0.877 * (pixel_in[2].x - Y[2]);
    table_id[2] = get_sector_id(ui[2], vi[2]);
    uo[2] = ui[2] * table[4 * table_id[2]] + vi[2] * table[4 * table_id[2] + 1];
    vo[2] = ui[2] * table[4 * table_id[2] + 2] + vi[2] * table[4 * table_id[2] + 3];
    pixel_out[2].x = Y[2] + 1.14 * vo[2];
    pixel_out[2].y = Y[2] - 0.39 * uo[2] - 0.58 * vo[1];
    pixel_out[2].z = Y[2] + 2.03 * uo[2];
    pixel_out[2].w = 0.0;
    write_imagef(output, (int2)(4*x+2,2*y), pixel_out[2]);

    Y[3] = 0.3 * pixel_in[3].x + 0.59 * pixel_in[3].y + 0.11 * pixel_in[3].z;
    ui[3] = 0.493 * (pixel_in[3].z - Y[3]);
    vi[3] = 0.877 * (pixel_in[3].x - Y[3]);
    table_id[3] = get_sector_id(ui[3], vi[3]);
    uo[3] = ui[3] * table[4 * table_id[3]] + vi[3] * table[4 * table_id[3] + 1];
    vo[3] = ui[3] * table[4 * table_id[3] + 2] + vi[3] * table[4 * table_id[3] + 3];
    pixel_out[3].x = Y[3] + 1.14 * vo[3];
    pixel_out[3].y = Y[3] - 0.39 * uo[3] - 0.58 * vo[3];
    pixel_out[3].z = Y[3] + 2.03 * uo[3];
    pixel_out[3].w = 0.0;
    write_imagef(output, (int2)(4*x+3,2*y), pixel_out[3]);

    Y[4] = 0.3 * pixel_in[4].x + 0.59 * pixel_in[4].y + 0.11 * pixel_in[4].z;
    ui[4] = 0.493 * (pixel_in[4].z - Y[4]);
    vi[4] = 0.877 * (pixel_in[4].x - Y[4]);
    table_id[4] = get_sector_id(ui[4], vi[4]);
    uo[4] = ui[4] * table[4 * table_id[4]] + vi[4] * table[4 * table_id[4] + 1];
    vo[4] = ui[4] * table[4 * table_id[4] + 2] + vi[4] * table[4 * table_id[4] + 3];
    pixel_out[4].x = Y[4] + 1.14 * vo[4];
    pixel_out[4].y = Y[4] - 0.39 * uo[4] - 0.58 * vo[4];
    pixel_out[4].z = Y[4] + 2.03 * uo[4];
    pixel_out[4].w = 0.0;
    write_imagef(output, (int2)(4*x,2*y+1), pixel_out[4]);

    Y[5] = 0.3 * pixel_in[5].x + 0.59 * pixel_in[5].y + 0.11 * pixel_in[5].z;
    ui[5] = 0.493 * (pixel_in[5].z - Y[5]);
    vi[5] = 0.877 * (pixel_in[5].x - Y[5]);
    table_id[5] = get_sector_id(ui[5], vi[5]);
    uo[5] = ui[5] * table[4 * table_id[5]] + vi[5] * table[4 * table_id[5] + 1];
    vo[5] = ui[5] * table[4 * table_id[5] + 2] + vi[5] * table[4 * table_id[5] + 3];
    pixel_out[5].x = Y[5] + 1.14 * vo[5];
    pixel_out[5].y = Y[5] - 0.39 * uo[5] - 0.58 * vo[5];
    pixel_out[5].z = Y[5] + 2.03 * uo[5];
    pixel_out[5].w = 0.0;
    write_imagef(output, (int2)(4*x+1,2*y+1), pixel_out[5]);

    Y[6] = 0.3 * pixel_in[6].x + 0.59 * pixel_in[6].y + 0.11 * pixel_in[6].z;
    ui[6] = 0.493 * (pixel_in[6].z - Y[6]);
    vi[6] = 0.877 * (pixel_in[6].x - Y[6]);
    table_id[6] = get_sector_id(ui[6], vi[6]);
    uo[6] = ui[6] * table[4 * table_id[6]] + vi[6] * table[4 * table_id[6] + 1];
    vo[6] = ui[6] * table[4 * table_id[6] + 2] + vi[6] * table[4 * table_id[6] + 3];
    pixel_out[6].x = Y[6] + 1.14 * vo[6];
    pixel_out[6].y = Y[6] - 0.39 * uo[6] - 0.58 * vo[6];
    pixel_out[6].z = Y[6] + 2.03 * uo[6];
    pixel_out[6].w = 0.0;
    write_imagef(output, (int2)(4*x+2,2*y+1), pixel_out[6]);

    Y[7] = 0.3 * pixel_in[7].x + 0.59 * pixel_in[7].y + 0.11 * pixel_in[7].z;
    ui[7] = 0.493 * (pixel_in[7].z - Y[7]);
    vi[7] = 0.877 * (pixel_in[7].x - Y[7]);
    table_id[7] = get_sector_id(ui[7], vi[7]);
    uo[7] = ui[7] * table[4 * table_id[7]] + vi[7] * table[4 * table_id[7] + 1];
    vo[7] = ui[7] * table[4 * table_id[7] + 2] + vi[7] * table[4 * table_id[7] + 3];
    pixel_out[7].x = Y[7] + 1.14 * vo[7];
    pixel_out[7].y = Y[7] - 0.39 * uo[7] - 0.58 * vo[7];
    pixel_out[7].z = Y[7] + 2.03 * uo[7];
    pixel_out[7].w = 0.0;
    write_imagef(output, (int2)(4*x+3,2*y+1), pixel_out[7]);

}


