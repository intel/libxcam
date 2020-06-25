/*
 * test-soft-image.cpp - test soft image
 *
 *  Copyright (c) 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "test_common.h"
#include "test_inline.h"
#include "test_stream.h"
#include "test_sv_params.h"

#include <soft/soft_video_buf_allocator.h>
#include <interface/blender.h>
#include <interface/geo_mapper.h>
#include <interface/stitcher.h>
#include <fisheye_dewarp.h>

#define MAP_WIDTH 3
#define MAP_HEIGHT 4

#define MAP_WIDTH_4K 15
#define MAP_HEIGHT_4K 12

static PointFloat2 map_table[MAP_HEIGHT * MAP_WIDTH] = {
    {160.0f, 120.0f}, {480.0f, 120.0f}, {796.0f, 120.0f},
    {60.0f, 240.0f}, {480.0f, 240.0f}, {900.0f, 240.0f},
    {16.0f, 360.0f}, {480.0f, 360.0f}, {944.0f, 360.0f},
    {0.0f, 480.0f}, {480.0f, 480.0f}, {960.0f, 480.0f},
};

static PointFloat2 map_table_insta[MAP_HEIGHT_4K * MAP_WIDTH_4K] = {
    { 3705.208496, 1477.394775 },
    { 3605.299316, 1135.900757 },
    { 3445.429688, 831.426636 },
    { 3236.703613, 570.348572 },
    { 2989.440430, 356.766235 },
    { 2713.069336, 193.226929 },
    { 2416.212891, 81.261719 },
    { 2106.875000, 21.755737 },
    { 1792.678467, 15.174194 },
    { 1481.120117, 61.674438 },
    { 1179.835327, 161.110474 },
    { 896.858643, 312.932983 },
    { 640.868530, 515.973938 },
    { 421.392700, 768.082153 },
    { 248.913208, 1065.594971 },
    { 3705.208496, 1477.394653 },
    { 3565.627441, 1172.575928 },
    { 3380.209961, 911.826050 },
    { 3160.185059, 695.461304 },
    { 2914.459717, 522.819580 },
    { 2650.178467, 392.926758 },
    { 2373.224365, 304.866821 },
    { 2088.643311, 257.977539 },
    { 1801.002441, 251.952393 },
    { 1514.707520, 286.883545 },
    { 1234.308594, 363.268799 },
    { 964.820129, 481.980591 },
    { 712.080566, 644.181152 },
    { 483.189087, 851.142334 },
    { 287.014526, 1103.899292 },
    { 3705.208496, 1477.394653 },
    { 3531.979492, 1219.865601 },
    { 3327.126953, 1006.901428 },
    { 3099.816650, 834.547058 },
    { 2856.572021, 699.425781 },
    { 2602.183594, 598.885254 },
    { 2340.324707, 531.002014 },
    { 2073.978516, 494.541565 },
    { 1805.748291, 488.923035 },
    { 1538.108765, 514.197998 },
    { 1273.636719, 571.056396 },
    { 1015.261536, 660.855957 },
    { 766.571777, 785.672913 },
    { 532.243774, 948.357117 },
    { 318.652588, 1152.556519 },
    { 3705.208496, 1477.394653 },
    { 3504.978760, 1275.607178 },
    { 3285.836426, 1112.964966 },
    { 3053.866211, 983.614868 },
    { 2813.078369, 883.316406 },
    { 2566.244873, 809.066711 },
    { 2315.390137, 758.826965 },
    { 2062.095215, 731.338318 },
    { 1807.705078, 726.009644 },
    { 1553.481445, 742.867432 },
    { 1300.742554, 782.558411 },
    { 1051.016602, 846.406616 },
    { 806.239746, 936.534180 },
    { 569.056885, 1056.056763 },
    { 343.295410, 1209.379883 },
    { 3705.208496, 1477.394653 },
    { 3485.027832, 1337.664063 },
    { 3255.942871, 1226.865234 },
    { 3020.991699, 1139.566284 },
    { 2782.077148, 1072.115479 },
    { 2540.462158, 1022.055664 },
    { 2297.040283, 987.774170 },
    { 2052.490723, 968.294739 },
    { 1807.378540, 963.160339 },
    { 1562.227661, 972.382202 },
    { 1317.583496, 996.440186 },
    { 1074.084717, 1036.338379 },
    { 832.554810, 1093.725464 },
    { 594.154175, 1171.109009 },
    { 360.630371, 1272.217651 },
    { 3705.208496, 1477.394531 },
    { 3472.378906, 1403.946533 },
    { 3237.115723, 1345.848877 },
    { 3000.257813, 1299.891113 },
    { 2762.317383, 1263.996216 },
    { 2523.626221, 1236.802734 },
    { 2284.414795, 1217.432373 },
    { 2044.853638, 1205.359131 },
    { 1805.081787, 1200.336548 },
    { 1565.225098, 1202.367676 },
    { 1325.412231, 1211.703979 },
    { 1085.791992, 1228.878296 },
    { 846.556885, 1254.777832 },
    { 607.982666, 1290.776733 },
    { 370.492554, 1338.971802 },
    { 3705.208496, 1477.394531 },
    { 3467.180908, 1472.408569 },
    { 3229.153320, 1467.422607 },
    { 2991.125000, 1462.436646 },
    { 2753.097412, 1457.450684 },
    { 2515.069580, 1452.464722 },
    { 2277.041992, 1447.478882 },
    { 2039.013672, 1442.492920 },
    { 1800.986328, 1437.506958 },
    { 1562.958374, 1432.520996 },
    { 1324.930420, 1427.535034 },
    { 1086.902832, 1422.549194 },
    { 848.874878, 1417.563232 },
    { 610.847290, 1412.577271 },
    { 372.819336, 1407.591431 },
    { 3705.208496, 1477.394531 },
    { 3469.507324, 1541.028198 },
    { 3232.017578, 1589.223145 },
    { 2993.442871, 1625.222046 },
    { 2754.208252, 1651.121582 },
    { 2514.587891, 1668.295898 },
    { 2274.774902, 1677.632080 },
    { 2034.918335, 1679.663086 },
    { 1795.146606, 1674.640625 },
    { 1555.585449, 1662.567261 },
    { 1316.373779, 1643.197144 },
    { 1077.683105, 1616.003662 },
    { 839.742310, 1580.108765 },
    { 602.884766, 1534.151123 },
    { 367.621216, 1476.053345 },
    { 3705.208496, 1477.394409 },
    { 3479.369629, 1607.782227 },
    { 3245.846191, 1708.890747 },
    { 3007.445313, 1786.274414 },
    { 2765.915527, 1843.661499 },
    { 2522.416504, 1883.559570 },
    { 2277.772461, 1907.617676 },
    { 2032.621582, 1916.839478 },
    { 1787.509521, 1911.705200 },
    { 1542.959839, 1892.225586 },
    { 1299.537842, 1857.944092 },
    { 1057.923340, 1807.884399 },
    { 819.008301, 1740.433594 },
    { 584.057373, 1653.134766 },
    { 354.972168, 1542.335938 },
    { 3705.208496, 1477.394409 },
    { 3496.704590, 1670.620117 },
    { 3270.943359, 1823.942993 },
    { 3033.760254, 1943.465698 },
    { 2788.983887, 2033.593262 },
    { 2539.257568, 2097.441650 },
    { 2286.518799, 2137.132568 },
    { 2032.295166, 2153.990234 },
    { 1777.905029, 2148.661621 },
    { 1524.610229, 2121.172852 },
    { 1273.755005, 2070.933350 },
    { 1026.921875, 1996.683594 },
    { 786.133911, 1896.384888 },
    { 554.164063, 1767.035156 },
    { 335.021362, 1604.392822 },
    { 3705.208496, 1477.394409 },
    { 3521.347412, 1727.443481 },
    { 3307.756348, 1931.642578 },
    { 3073.428223, 2094.327148 },
    { 2824.738770, 2219.144043 },
    { 2566.363281, 2308.943359 },
    { 2301.891357, 2365.801758 },
    { 2034.251953, 2391.076660 },
    { 1766.021729, 2385.458252 },
    { 1499.675537, 2348.998047 },
    { 1237.816528, 2281.114502 },
    { 983.428284, 2180.574219 },
    { 740.183472, 2045.452881 },
    { 512.873535, 1873.098755 },
    { 308.020508, 1660.134399 },
    { 3705.208496, 1477.394287 },
    { 3552.985352, 1776.100586 },
    { 3356.811523, 2028.857422 },
    { 3127.919434, 2235.818848 },
    { 2875.180176, 2398.019287 },
    { 2605.691162, 2516.730957 },
    { 2325.292725, 2593.116211 },
    { 2038.997803, 2628.047363 },
    { 1751.356934, 2622.022217 },
    { 1466.775879, 2575.132813 },
    { 1189.821777, 2487.072998 },
    { 925.540649, 2357.180420 },
    { 679.815063, 2184.538574 },
    { 459.790771, 1968.174072 },
    { 274.372681, 1707.424072 },
};

using namespace XCam;

enum SoftType {
    SoftTypeNone    = 0,
    SoftTypeBlender,
    SoftTypeRemap
};

#define TEST_MAP_FACTOR_X  16
#define TEST_MAP_FACTOR_Y  16

class SoftStream
    : public Stream
{
public:
    explicit SoftStream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~SoftStream () {}

    virtual XCamReturn create_buf_pool (uint32_t reserve_count, uint32_t format = V4L2_PIX_FMT_NV12);

private:
    XCAM_DEAD_COPY (SoftStream);
};
typedef std::vector<SmartPtr<SoftStream>> SoftStreams;

SoftStream::SoftStream (const char *file_name, uint32_t width, uint32_t height)
    :  Stream (file_name, width, height)
{
}

XCamReturn
SoftStream::create_buf_pool (uint32_t reserve_count, uint32_t format)
{
    XCAM_ASSERT (get_width () && get_height ());

    VideoBufferInfo info;
    info.init (format, get_width (), get_height ());

    SmartPtr<BufferPool> pool = new SoftVideoBufAllocator ();
    XCAM_ASSERT (pool.ptr ());
    if (!pool->set_video_info (info) || !pool->reserve (reserve_count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

void
set_map_table (SmartPtr<GeoMapper> mapper, uint32_t stitch_width, uint32_t stitch_height, CamModel cam_model)
{
    SmartPtr<SphereFisheyeDewarp> dewarper = new SphereFisheyeDewarp ();

    float vp_range[XCAM_STITCH_FISHEYE_MAX_NUM];
    viewpoints_range (cam_model, vp_range);

    StitchScopicMode scopic_mode = ScopicStereoRight;
    StitchInfo info = soft_stitch_info (cam_model, scopic_mode);
    dewarper->set_fisheye_info (info.fisheye_info[0]);

    Stitcher::RoundViewSlice view_slice;
    uint32_t alignment_x = 8;
    //uint32_t alignment_y = 4;
    view_slice.width = vp_range[0] / 360.0f * (float)stitch_width;
    view_slice.width = XCAM_ALIGN_UP (view_slice.width, alignment_x);
    view_slice.height = stitch_height;
    view_slice.hori_angle_range = view_slice.width * 360.0f / (float)stitch_width;

    float max_dst_latitude = (info.fisheye_info[0].intrinsic.fov > 180.0f) ? 180.0f : info.fisheye_info[0].intrinsic.fov;
    float max_dst_longitude = max_dst_latitude * view_slice.width / view_slice.height;

    dewarper->set_dst_range (max_dst_longitude, max_dst_latitude);
    dewarper->set_out_size (view_slice.width, view_slice.height);

    uint32_t table_width = view_slice.width / TEST_MAP_FACTOR_X;
    table_width = XCAM_ALIGN_UP (table_width, 4);
    uint32_t table_height = view_slice.height / TEST_MAP_FACTOR_Y;
    table_height = XCAM_ALIGN_UP (table_height, 2);
    dewarper->set_table_size (table_width, table_height);

    FisheyeDewarp::MapTable map_table (table_width * table_height);
    dewarper->gen_table (map_table);

    mapper->set_lookup_table (map_table.data (), table_width, table_height);
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --type TYPE --input0 input.nv12 --input1 input1.nv12 --output output.nv12 ...\n"
            "\t--type              processing type, selected from: blend, remap\n"
            "\t--cam-model          optional, camera model\n"
            "\t                    select from [cama2c1080p/camb4c1080p/camc3c8k/camd3c8k], default: camb4c1080p\n"
            "\t--input0            input image(NV12)\n"
            "\t--input1            input image(NV12)\n"
            "\t--output            output image(NV12/MP4)\n"
            "\t--in-w              optional, input width, default: 1280\n"
            "\t--in-h              optional, input height, default: 800\n"
            "\t--out-w             optional, output width, default: 1280\n"
            "\t--out-h             optional, output height, default: 800\n"
            "\t--save              optional, save file or not, select from [true/false], default: true\n"
            "\t--loop              optional, how many loops need to run, default: 1\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    uint32_t input_width = 1280;
    uint32_t input_height = 800;
    uint32_t output_width = 1280;
    uint32_t output_height = 800;

    uint32_t input_format = V4L2_PIX_FMT_NV12;
    CamModel cam_model = CamD3C8K;

    SoftStreams ins;
    SoftStreams outs;
    SoftType type = SoftTypeNone;

    int loop = 1;
    bool save_output = true;

    const struct option long_opts[] = {
        {"type", required_argument, NULL, 't'},
        {"input0", required_argument, NULL, 'i'},
        {"input1", required_argument, NULL, 'j'},
        {"in-format", required_argument, NULL, 'f'},
        {"cam-model", required_argument, NULL, 'C'},
        {"output", required_argument, NULL, 'o'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"save", required_argument, NULL, 's'},
        {"loop", required_argument, NULL, 'l'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 't':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "blend"))
                type = SoftTypeBlender;
            else if (!strcasecmp (optarg, "remap"))
                type = SoftTypeRemap;
            else {
                XCAM_LOG_ERROR ("unknown type:%s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'i':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SoftStream, ins, optarg);
            break;
        case 'j':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SoftStream, ins, optarg);
            break;
        case 'C':
            if (!strcasecmp (optarg, "cama2c1080p"))
                cam_model = CamA2C1080P;
            else if (!strcasecmp (optarg, "camb4c1080p"))
                cam_model = CamB4C1080P;
            else if (!strcasecmp (optarg, "camc3c8k"))
                cam_model = CamC3C8K;
            else if (!strcasecmp (optarg, "camd3c8k"))
                cam_model = CamD3C8K;
            else {
                XCAM_LOG_ERROR ("incorrect camera model: %s", optarg);
                usage (argv[0]);
                return -1;
            }
        case 'o':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SoftStream, outs, optarg);
            break;
        case 'f':
            input_format = (strcasecmp (optarg, "yuv") == 0 ? V4L2_PIX_FMT_YUV420 : V4L2_PIX_FMT_NV12);
            break;
        case 'w':
            input_width = atoi(optarg);
            break;
        case 'h':
            input_height = atoi(optarg);
            break;
        case 'W':
            output_width = atoi(optarg);
            break;
        case 'H':
            output_height = atoi(optarg);
            break;
        case 's':
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'l':
            loop = atoi(optarg);
            break;
        case 'e':
            usage (argv[0]);
            return 0;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value:%c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknown option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    if (ins.empty () || outs.empty () ||
            !strlen (ins[0]->get_file_name ()) || !strlen (outs[0]->get_file_name ())) {
        XCAM_LOG_ERROR ("input or output file name was not set");
        usage (argv[0]);
        return -1;
    }

    for (uint32_t i = 0; i < ins.size (); ++i) {
        printf ("input%d file:\t\t%s\n", i, ins[i]->get_file_name ());
    }
    printf ("output file:\t\t%s\n", outs[0]->get_file_name ());
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("save output:\t\t%s\n", save_output ? "true" : "false");
    printf ("loop count:\t\t%d\n", loop);

    XCAM_UNUSED (intrinsic_names);
    XCAM_UNUSED (extrinsic_names);

    for (uint32_t i = 0; i < ins.size (); ++i) {
        ins[i]->set_buf_size (input_width, input_height);
        CHECK (ins[i]->create_buf_pool (6, input_format), "create buffer pool failed");
        CHECK (ins[i]->open_reader ("rb"), "open input file(%s) failed", ins[i]->get_file_name ());
    }

    outs[0]->set_buf_size (output_width, output_height);
    if (save_output) {
        CHECK (outs[0]->estimate_file_format (), "%s: estimate file format failed", outs[0]->get_file_name ());
        CHECK (outs[0]->open_writer ("wb"), "open output file(%s) failed", outs[0]->get_file_name ());
    }

    switch (type) {
    case SoftTypeBlender: {
        CHECK_EXP (ins.size () == 2, "blender needs 2 input files.");
        SmartPtr<Blender> blender = Blender::create_soft_blender ();
        XCAM_ASSERT (blender.ptr ());
        blender->set_output_size (output_width, output_height);

        Rect area;
        area.pos_x = 0;
        area.pos_y = 0;
        area.width = output_width;
        area.height = output_height;
        blender->set_merge_window (area);
        area.pos_x = 0;
        area.pos_y = 0;
        area.width = input_width;
        area.height = input_height;
        blender->set_input_merge_area (area, 0);
        area.pos_x = 0;
        area.pos_y = 0;
        area.width = input_width;
        area.height = input_height;
        blender->set_input_merge_area (area, 1);

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        CHECK (ins[1]->read_buf(), "read buffer from file(%s) failed.", ins[1]->get_file_name ());
        for (int i = 0; i < loop; ++i) {
            CHECK (blender->blend (ins[0]->get_buf (), ins[1]->get_buf (), outs[0]->get_buf ()), "blend buffer failed");
            if (save_output)
                outs[0]->write_buf ();
            FPS_CALCULATION (soft_blend, XCAM_OBJ_DUR_FRAME_NUM);
        }
        break;
    }
    case SoftTypeRemap: {
        SmartPtr<GeoMapper> mapper = GeoMapper::create_soft_geo_mapper ();
        XCAM_ASSERT (mapper.ptr ());
        mapper->set_output_size (output_width, output_height);

#if 0
        if (input_width > 3800 && input_height > 2800) {
            mapper->set_lookup_table (map_table_insta, MAP_WIDTH_4K, MAP_HEIGHT_4K);
        } else {
            mapper->set_lookup_table (map_table, MAP_WIDTH, MAP_HEIGHT);
        }
#else
        uint32_t stitch_width = 7680;
        uint32_t stitch_height = 3840;
        set_map_table (mapper, stitch_width, stitch_height, cam_model);
#endif
        //mapper->set_factors ((output_width - 1.0f) / (MAP_WIDTH - 1.0f), (output_height - 1.0f) / (MAP_HEIGHT - 1.0f));

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        for (int i = 0; i < loop; ++i) {
            CHECK (mapper->remap (ins[0]->get_buf (), outs[0]->get_buf ()), "remap buffer failed");
            if (save_output)
                outs[0]->write_buf ();
            FPS_CALCULATION (soft_remap, XCAM_OBJ_DUR_FRAME_NUM);
        }
        break;
    }
    default: {
        XCAM_LOG_ERROR ("unsupported type:%d", type);
        usage (argv[0]);
        return -1;
    }
    }

    return 0;
}
