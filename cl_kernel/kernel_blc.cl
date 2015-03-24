/*
 * function: kernel_blc
 *     black level correction for sensor data input
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * blc_config: black level correction configuration 
 * param:    
 */

"typedef struct                                                                                "
"{                                                                                             "
"       float  level_gr;  /* Black level for GR pixels */                                      "
"	float  level_r;   /* Black level for R pixels */                                       "
"	float  level_b;   /* Black level for B pixels */                                       "
"	float  level_gb;   /* Black level for GB pixels */                                     "
"}BLCConfig;                                                                                   "
"                                                                                              "
"__kernel void kernel_blc (__read_only image2d_t input,                                        "
"                          __write_only image2d_t output,                                      "
"                          BLCConfig blc_config)                                               "
"{                                                                                             "
"    int x0 = 2 * get_global_id (0);                                                           "
"    int y0 = 2 * get_global_id (1);                                                           "
"    int x1 = x0 + 1;                                                                          "
"    int y1 = y0 + 1;                                                                          "
"    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;  "
"                                                                                              "
"    int2 pos_r = (int2)(x0, y0);                                                              "
"    int2 pos_gr = (int2)(x1, y0);                                                             "
"    int2 pos_gb = (int2)(x0, y1);                                                             "
"    int2 pos_b = (int2)(x1, y1);                                                              "
"                                                                                              "
"    uint4 pixel;                                                                              "
"    pixel = read_imageui(input, sampler, pos_r);                                              "
"    pixel.x = pixel.x << 6;                                                                   "
"    pixel.x = pixel.x - blc_config.level_r * 65536;                                           "
"    write_imageui(output, pos_r, pixel);                                                      "
"                                                                                              "
"    pixel = read_imageui(input, sampler, pos_gr);                                             "
"    pixel.x = pixel.x << 6;                                                                   "
"    pixel.x = pixel.x - blc_config.level_gr * 65536;                                          "
"    write_imageui(output, pos_gr, pixel);                                                     "
"                                                                                              "
"    pixel = read_imageui(input, sampler, pos_gb);                                             "
"    pixel.x = pixel.x << 6;                                                                   "
"    pixel.x = pixel.x - blc_config.level_gb * 65536;                                          "
"    write_imageui(output, pos_gb, pixel);                                                     "
"                                                                                              "
"    pixel = read_imageui(input, sampler, pos_b);                                              "
"    pixel.x = pixel.x << 6;                                                                   "
"    pixel.x = pixel.x - blc_config.level_b * 65536;                                           "
"    write_imageui(output, pos_b, pixel);                                                      "
"}                                                                                             " 

