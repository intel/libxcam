/*
 * test-image-stitching.cpp - test image stitching
 *
 *  Copyright (c) 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "test_common.h"
#include "test_inline.h"
#include "test_stream.h"
#include <image_file_handle.h>
#include <ocl/cl_device.h>
#include <ocl/cl_context.h>
#include <ocl/cl_fisheye_handler.h>
#include <ocl/cl_image_360_stitch.h>
#include <ocl/cl_utils.h>
#if HAVE_OPENCV
#include "ocv/cv_utils.h"
#endif

using namespace XCam;

#define XCAM_TEST_STITCH_DEBUG 0
#define XCAM_ALIGNED_WIDTH 16

#if HAVE_OPENCV
#define FOURCC_X264 cv::VideoWriter::fourcc ('X', '2', '6', '4')
#endif

#define CHECK_ACCESS(fliename) \
    if (access (fliename, F_OK) != 0) {            \
        XCAM_LOG_ERROR ("%s not found", fliename); \
        return false;                              \
    }

enum SVOutIdx {
    IdxStitch    = 0,
    IdxTopView,
    IdxFreeView,
    IdxCount
};

static const char *intrinsic_names[] = {
    "intrinsic_camera_front.txt",
    "intrinsic_camera_right.txt",
    "intrinsic_camera_rear.txt",
    "intrinsic_camera_left.txt"
};
static const char *extrinsic_names[] = {
    "extrinsic_camera_front.txt",
    "extrinsic_camera_right.txt",
    "extrinsic_camera_rear.txt",
    "extrinsic_camera_left.txt"
};

static const float viewpoints_range[] = {64.0f, 160.0f, 64.0f, 160.0f};

class CLStream
    : public Stream
{
public:
    explicit CLStream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~CLStream () {}

    virtual XCamReturn create_buf_pool (uint32_t reserve_count, uint32_t format = V4L2_PIX_FMT_NV12);
};
typedef std::vector<SmartPtr<CLStream>> CLStreams;

CLStream::CLStream (const char *file_name, uint32_t width, uint32_t height)
    : Stream (file_name, width, height)
{
}

XCamReturn
CLStream::create_buf_pool (uint32_t reserve_count, uint32_t format)
{
    XCAM_ASSERT (get_width () && get_height ());

    VideoBufferInfo info;
    info.init (format, get_width (), get_height ());

    SmartPtr<CLVideoBufferPool> pool = new CLVideoBufferPool ();
    XCAM_ASSERT (pool.ptr ());
    if (!pool->set_video_info (info) || !pool->reserve (reserve_count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);

    return XCAM_RETURN_NO_ERROR;
}

static void
combine_name (const char *orig_name, const char *embedded_str, char *new_name)
{
    const char *dir_delimiter = strrchr (orig_name, '/');

    if (dir_delimiter) {
        std::string path (orig_name, dir_delimiter - orig_name + 1);
        XCAM_ASSERT (path.c_str ());
        snprintf (new_name, XCAM_TEST_MAX_STR_SIZE, "%s%s_%s", path.c_str (), embedded_str, dir_delimiter + 1);
    } else {
        snprintf (new_name, XCAM_TEST_MAX_STR_SIZE, "%s_%s", embedded_str, orig_name);
    }
}

static void
add_stream (CLStreams &streams, const char *stream_name, uint32_t width, uint32_t height)
{
    char file_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    combine_name (streams[0]->get_file_name (), stream_name, file_name);

    SmartPtr<CLStream> stream = new CLStream (file_name, width, height);
    XCAM_ASSERT (stream.ptr ());
    streams.push_back (stream);
}

static void
write_in_image (
    const SmartPtr<CLImage360Stitch> &stitcher, const CLStreams &ins, uint32_t frame_num)
{
#if (XCAM_TEST_STREAM_DEBUG) && (XCAM_TEST_OPENCV)
    char img_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    char frame_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    std::snprintf (frame_str, XCAM_TEST_MAX_STR_SIZE, "frame:%d", frame_num);

    StitchInfo info = stitcher->get_stitch_info ();

    cv::Mat mat;
    if (ins.size () == 1) {
        convert_to_mat (ins[0]->get_buf (), mat);

        for (int i = 0; i < stitcher->get_fisheye_num (); i++) {
            cv::circle (mat, cv::Point (info.fisheye_info[i].intrinsic.cx, info.fisheye_info[i].intrinsic.cy),
                        info.fisheye_info[i].radius, cv::Scalar(0, 0, 255), 2);
        }
        cv::putText (mat, frame_str, cv::Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 2.0,
                     cv::Scalar(0, 0, 255), 2, 8, false);

        std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "orig_fisheye_%d.jpg", frame_num);
        cv::imwrite (img_name, mat);
    } else {
        char idx_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
        for (uint32_t i = 0; i < ins.size (); i++) {
            convert_to_mat (ins[i]->get_buf (), mat);

            cv::circle (mat, cv::Point (info.fisheye_info[i].intrinsic.cx, info.fisheye_info[i].intrinsic.cy),
                        info.fisheye_info[i].radius, cv::Scalar(0, 0, 255), 2);
            cv::putText (mat, frame_str, cv::Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 2.0,
                         cv::Scalar(0, 0, 255), 2, 8, false);

            std::snprintf (idx_str, XCAM_TEST_MAX_STR_SIZE, "idx:%d", i);
            cv::putText (mat, idx_str, cv::Point (20, 110), cv::FONT_HERSHEY_COMPLEX, 2.0,
                         cv::Scalar (0, 0, 255), 2, 8, false);

            std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "orig_fisheye_%d_%d.jpg", frame_num, i);
            cv::imwrite (img_name, mat);
        }
    }
#else
    XCAM_UNUSED (stitcher);
    XCAM_UNUSED (ins);
    XCAM_UNUSED (frame_num);
#endif
}

static void
write_out_image (const SmartPtr<CLStream> &out, uint32_t frame_num)
{
#if !XCAM_TEST_STREAM_DEBUG
    XCAM_UNUSED (frame_num);
    out->write_buf ();
#else
    char frame_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    std::snprintf (frame_str, XCAM_TEST_MAX_STR_SIZE, "frame:%d", frame_num);
    out->write_buf (frame_str);

#if XCAM_TEST_OPENCV
    char img_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "%s_%d.jpg", out->get_file_name (), frame_num);
    out->debug_write_image (img_name, frame_str);
#endif
#endif
}

static void
write_image (
    const SmartPtr<CLImage360Stitch> &stitcher,
    const CLStreams &ins, const CLStreams &outs,
    bool save_output, bool save_topview, bool save_freeview)
{
    static uint32_t frame_num = 0;

    write_in_image (stitcher, ins, frame_num);

    if (save_output)
        write_out_image (outs[IdxStitch], frame_num);

    const BowlDataConfig config = stitcher->get_fisheye_bowl_config ();
    if (save_topview) {
        std::vector<PointFloat2> map_table;

        XCAM_ASSERT (outs[IdxTopView]->get_buf ().ptr ());
        sample_generate_top_view (
            outs[IdxStitch]->get_buf (), outs[IdxTopView]->get_buf (), config, map_table);
        write_out_image (outs[IdxTopView], frame_num);
    }

    if (save_freeview) {
        std::vector<PointFloat2> map_table;
        float start_angle = -45.0f, end_angle = 45.0f;

        XCAM_ASSERT (outs[IdxFreeView]->get_buf ().ptr ());
        sample_generate_rectified_view (
            outs[IdxStitch]->get_buf (), outs[IdxFreeView]->get_buf (),
            config, start_angle, end_angle, map_table);
        write_out_image (outs[IdxFreeView], frame_num);
    }

    frame_num++;
}

static int
run_stitcher (
    const SmartPtr<CLImage360Stitch> &stitcher,
    const CLStreams &ins, const CLStreams &outs,
    bool save_output, bool save_topview, bool save_freeview, int loop)
{
    CHECK (check_streams<CLStreams> (ins), "invalid input streams");
    CHECK (check_streams<CLStreams> (outs), "invalid output streams");

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<VideoBuffer> in_buffers, pre_buf;
    while (loop--) {
        for (uint32_t i = 0; i < ins.size (); ++i) {
            CHECK (ins[i]->rewind (), "rewind buffer from file(%s) failed", ins[i]->get_file_name ());
        }

        do {
            for (uint32_t i = 0; i < ins.size (); ++i) {
                ret = ins[i]->read_buf();
                if (ret == XCAM_RETURN_BYPASS)
                    break;
                CHECK (ret, "read buffer from file(%s) failed", ins[i]->get_file_name ());

                if (i == 0)
                    in_buffers = ins[i]->get_buf ();
                else
                    pre_buf->attach_buffer (ins[i]->get_buf ());

                pre_buf = ins[i]->get_buf ();
            }
            if (ret == XCAM_RETURN_BYPASS)
                break;

            ret = stitcher->execute (in_buffers, outs[IdxStitch]->get_buf ());
            CHECK (ret, "execute stitcher failed");

            if (save_output || save_topview || save_freeview)
                write_image (stitcher, ins, outs, save_output, save_topview, save_freeview);

            FPS_CALCULATION (image_stitching, XCAM_OBJ_DUR_FRAME_NUM);
        } while (true);
    }

    return 0;
}

void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input file --output file\n"
            "\t--input             input image(NV12)\n"
            "\t--output            output image(NV12)\n"
            "\t--input-w           optional, input width, default: 1920\n"
            "\t--input-h           optional, input height, default: 1080\n"
            "\t--output-w          optional, output width, default: 1920\n"
            "\t--output-h          optional, output width, default: 960\n"
            "\t--res-mode          optional, image resolution mode\n"
            "\t                    select from [1080p2cams/1080p4cams/4k2cams/8k6cams], default: 1080p2cams\n"
            "\t--dewarp-mode       optional, fisheye dewarp mode, select from [sphere, bowl], default: sphere\n"
            "\t--scale-mode        optional, image scaling mode, select from [local/global], default: local\n"
            "\t--enable-seam       optional, enable seam finder in blending area, default: no\n"
            "\t--enable-fisheyemap optional, enable fisheye map, default: no\n"
            "\t--enable-lsc        optional, enable lens shading correction, default: no\n"
#if HAVE_OPENCV
            "\t--fm                optional, enable or disable feature match, default: true\n"
#endif
            "\t--fisheye-num       optional, the number of fisheye lens, default: 2\n"
            "\t--save              optional, save file or not, select from [true/false], default: true\n"
            "\t--save-topview      optional, save top view videos, select from [true/false], default: false\n"
            "\t--save-freeview     optional, save free(rectified) view videos, select from [true/false], default: false\n"
            "\t--framerate         optional, framerate of saved video, default: 30.0\n"
            "\t--loop              optional, how many loops need to run for performance test, default: 1\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t output_height = 960;
    uint32_t output_width = output_height * 2;
    uint32_t topview_width = 1920;
    uint32_t topview_height = 1080;
    uint32_t freeview_width = 1920;
    uint32_t freeview_height = 1080;

    CLStreams ins;
    CLStreams outs;

    int loop = 1;
    bool enable_seam = false;
    bool enable_fisheye_map = false;
    bool enable_lsc = false;
#if HAVE_OPENCV
    bool need_fm = true;
#endif
    CLBlenderScaleMode scale_mode = CLBlenderScaleLocal;
    StitchResMode res_mode = StitchRes1080P2Cams;
    FisheyeDewarpMode dewarp_mode = DewarpSphere;

    int fisheye_num = 2;
    bool save_output = true;
    bool save_topview = false;
    bool save_freeview = false;
    double framerate = 30.0;

    const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"input-w", required_argument, NULL, 'w'},
        {"input-h", required_argument, NULL, 'h'},
        {"output-w", required_argument, NULL, 'W'},
        {"output-h", required_argument, NULL, 'H'},
        {"res-mode", required_argument, NULL, 'R'},
        {"dewarp-mode", required_argument, NULL, 'r'},
        {"scale-mode", required_argument, NULL, 'c'},
        {"enable-seam", no_argument, NULL, 'S'},
        {"enable-fisheyemap", no_argument, NULL, 'F'},
        {"enable-lsc", no_argument, NULL, 'L'},
#if HAVE_OPENCV
        {"fm", required_argument, NULL, 'm'},
#endif
        {"fisheye-num", required_argument, NULL, 'N'},
        {"all-in-one", required_argument, NULL, 'A'},
        {"save", required_argument, NULL, 's'},
        {"save-topview", required_argument, NULL, 't'},
        {"save-freeview", required_argument, NULL, 'v'},
        {"framerate", required_argument, NULL, 'f'},
        {"loop", required_argument, NULL, 'l'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (CLStream, ins, optarg);
            break;
        case 'o':
            PUSH_STREAM (CLStream, outs, optarg);
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
        case 'R':
            if (!strcasecmp (optarg, "1080p2cams"))
                res_mode = StitchRes1080P2Cams;
            else if (!strcasecmp (optarg, "1080p4cams"))
                res_mode = StitchRes1080P4Cams;
            else if (!strcasecmp (optarg, "4k2cams"))
                res_mode = StitchRes4K2Cams;
            else if (!strcasecmp (optarg, "8k6cams"))
                res_mode = StitchRes8K6Cams;
            else {
                XCAM_LOG_ERROR ("incorrect resolution mode");
                return -1;
            }
            break;
        case 'r':
            if (!strcasecmp (optarg, "sphere"))
                dewarp_mode = DewarpSphere;
            else if(!strcasecmp (optarg, "bowl"))
                dewarp_mode = DewarpBowl;
            else {
                XCAM_LOG_ERROR ("incorrect fisheye dewarp mode");
                return -1;
            }
            break;
        case 'c':
            if (!strcasecmp (optarg, "local"))
                scale_mode = CLBlenderScaleLocal;
            else if (!strcasecmp (optarg, "global"))
                scale_mode = CLBlenderScaleGlobal;
            else {
                XCAM_LOG_ERROR ("incorrect scaling mode");
                return -1;
            }
            break;
        case 'S':
            enable_seam = true;
            break;
        case 'F':
            enable_fisheye_map = true;
            break;
        case 'L':
            enable_lsc = true;
            break;
#if HAVE_OPENCV
        case 'm':
            need_fm = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
#endif
        case 'N':
            fisheye_num = atoi(optarg);
            if (fisheye_num > XCAM_STITCH_FISHEYE_MAX_NUM) {
                XCAM_LOG_ERROR ("fisheye number should not be greater than %d\n", XCAM_STITCH_FISHEYE_MAX_NUM);
                return -1;
            }
            break;
        case 's':
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 't':
            save_topview = (strcasecmp (optarg, "true") == 0 ? true : false);
            break;
        case 'v':
            save_freeview = (strcasecmp (optarg, "true") == 0 ? true : false);
            break;
        case 'f':
            framerate = atof(optarg);
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

    if ((ins.size () != 1) && (ins.size () != (uint32_t)fisheye_num)) {
        XCAM_LOG_ERROR (
            "multiple-input mode: conflicting input number(%d) and fisheye number(%d)",
            ins.size (), fisheye_num);
        return -1;
    }

    for (uint32_t i = 0; i < ins.size (); ++i) {
        CHECK_EXP (ins[i].ptr (), "input stream is NULL, index:%d", i);
        CHECK_EXP (strlen (ins[i]->get_file_name ()), "input file name was not set, index:%d", i);
    }

    CHECK_EXP (outs.size () == 1 && outs[IdxStitch].ptr (), "surrond view needs 1 output stream");
    CHECK_EXP (strlen (outs[IdxStitch]->get_file_name ()), "output file name was not set");

    output_width = XCAM_ALIGN_UP (output_width, XCAM_ALIGNED_WIDTH);
    output_height = XCAM_ALIGN_UP (output_height, XCAM_ALIGNED_WIDTH);

    for (uint32_t i = 0; i < ins.size (); ++i) {
        printf ("input%d file:\t\t%s\n", i, ins[i]->get_file_name ());
    }
    printf ("output file:\t\t%s\n", outs[IdxStitch]->get_file_name ());
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("resolution mode:\t%s\n", res_mode == StitchRes1080P2Cams ? "1080p2cams" :
            (res_mode == StitchRes1080P4Cams ? "1080p4cams" : (res_mode == StitchRes4K2Cams ? "4k2cams" : "8k6cams")));
    printf ("fisheye dewarp mode: \t%s\n", dewarp_mode == DewarpSphere ? "sphere" : "bowl");
    printf ("scale mode:\t\t%s\n", scale_mode == CLBlenderScaleLocal ? "local" : "global");
    printf ("seam mask:\t\t%s\n", enable_seam ? "true" : "false");
    printf ("fisheye map:\t\t%s\n", enable_fisheye_map ? "true" : "false");
    printf ("shading correction:\t%s\n", enable_lsc ? "true" : "false");
#if HAVE_OPENCV
    printf ("feature match:\t\t%s\n", need_fm ? "true" : "false");
#endif
    printf ("fisheye number:\t\t%d\n", fisheye_num);
    printf ("save file:\t\t%s\n", save_output ? "true" : "false");
    printf ("save topview file:\t%s\n", save_topview ? "true" : "false");
    printf ("save freeview file:\t%s\n", save_freeview ? "true" : "false");
    printf ("framerate:\t\t%.3lf\n", framerate);
    printf ("loop count:\t\t%d\n", loop);
    printf ("-----------------------------------\n");

    for (uint32_t i = 0; i < ins.size (); ++i) {
        ins[i]->set_buf_size (input_width, input_height);
        CHECK (ins[i]->create_buf_pool (6), "create buffer pool failed");
        CHECK (ins[i]->open_reader ("rb"), "open input file(%s) failed", ins[i]->get_file_name ());
    }

    outs[IdxStitch]->set_buf_size (output_width, output_height);
    if (save_output) {
        CHECK (outs[IdxStitch]->estimate_file_format (),
               "%s: estimate file format failed", outs[IdxStitch]->get_file_name ());
        CHECK (outs[IdxStitch]->open_writer ("wb"), "open output file(%s) failed", outs[IdxStitch]->get_file_name ());
    }

    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    SmartPtr<CLImage360Stitch> stitcher = create_image_360_stitch (
            context, enable_seam, scale_mode, enable_fisheye_map, enable_lsc,
            dewarp_mode, res_mode, fisheye_num).dynamic_cast_ptr<CLImage360Stitch> ();
    XCAM_ASSERT (stitcher.ptr ());
    stitcher->set_output_size (output_width, output_height);
    stitcher->set_pool_type (CLImageHandler::CLVideoPoolType);
#if HAVE_OPENCV
    stitcher->set_feature_match (need_fm);
#endif

    if (dewarp_mode == DewarpBowl) {
        stitcher->set_intrinsic_names (intrinsic_names);
        stitcher->set_extrinsic_names (extrinsic_names);
    }

    add_stream (outs, "topview", topview_width, topview_height);
    if (save_topview) {
        CHECK (outs[IdxTopView]->create_buf_pool (1), "create topview buffer pool failed");
        CHECK (outs[IdxTopView]->estimate_file_format (),
               "%s: estimate file format failed", outs[IdxTopView]->get_file_name ());
        CHECK (outs[IdxTopView]->open_writer ("wb"),
               "open topview file(%s) failed", outs[IdxTopView]->get_file_name ());
    }

    add_stream (outs, "freeview", freeview_width, freeview_height);
    if (save_freeview) {
        CHECK (outs[IdxFreeView]->create_buf_pool (1), "create freeview buffer pool failed");
        CHECK (outs[IdxFreeView]->estimate_file_format (),
               "%s: estimate file format failed", outs[IdxFreeView]->get_file_name ());
        CHECK (outs[IdxFreeView]->open_writer ("wb"),
               "open freeview file(%s) failed", outs[IdxFreeView]->get_file_name ());
    }

    CHECK_EXP (
        run_stitcher (stitcher, ins, outs, save_output, save_topview, save_freeview, loop) == 0,
        "run stitcher failed");

    return 0;
}

