/*
 * test-surround-view.cpp - test surround view
 *
 *  Copyright (c) 2018 Intel Corporation
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
 */

#include "test_common.h"
#include "test_stream.h"
#include <interface/geo_mapper.h>
#include <interface/stitcher.h>
#include <calibration_parser.h>
#include <soft/soft_video_buf_allocator.h>
#if HAVE_GLES
#include <gles/gl_video_buffer.h>
#include <gles/egl/egl_base.h>
#endif
#if HAVE_VULKAN
#include <vulkan/vk_device.h>
#endif

using namespace XCam;

enum FrameMode {
    FrameSingle = 0,
    FrameMulti
};

enum SVModule {
    SVModuleNone    = 0,
    SVModuleSoft,
    SVModuleGLES,
    SVModuleVulkan
};

enum SVOutIdx {
    IdxStitch    = 0,
    IdxTopView,
    IdxCount
};

static const char *instrinsic_names[] = {
    "intrinsic_camera_front.txt",
    "intrinsic_camera_right.txt",
    "intrinsic_camera_rear.txt",
    "intrinsic_camera_left.txt"
};

static const char *exstrinsic_names[] = {
    "extrinsic_camera_front.txt",
    "extrinsic_camera_right.txt",
    "extrinsic_camera_rear.txt",
    "extrinsic_camera_left.txt"
};

class SVStream
    : public Stream
{
public:
    explicit SVStream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~SVStream () {}

    void set_module (SVModule module) {
        XCAM_ASSERT (module != SVModuleNone);
        _module = module;
    }

    void set_mapper (const SmartPtr<GeoMapper> &mapper) {
        XCAM_ASSERT (mapper.ptr ());
        _mapper = mapper;
    }
    const SmartPtr<GeoMapper> &get_mapper () {
        return _mapper;
    }

#if HAVE_VULKAN
    void set_vk_device (SmartPtr<VKDevice> &device) {
        XCAM_ASSERT (device.ptr ());
        _vk_dev = device;
    }
    SmartPtr<VKDevice> &get_vk_device () {
        return _vk_dev;
    }
#endif

    virtual XCamReturn create_buf_pool (uint32_t reserve_count);

private:
    XCAM_DEAD_COPY (SVStream);

private:
    SVModule               _module;
    SmartPtr<GeoMapper>    _mapper;
#if HAVE_VULKAN
    SmartPtr<VKDevice>     _vk_dev;
#endif
};
typedef std::vector<SmartPtr<SVStream>> SVStreams;

SVStream::SVStream (const char *file_name, uint32_t width, uint32_t height)
    :  Stream (file_name, width, height)
    , _module (SVModuleNone)
{
}

XCamReturn
SVStream::create_buf_pool (uint32_t reserve_count)
{
    XCAM_ASSERT (get_width () && get_height ());
    XCAM_FAIL_RETURN (
        ERROR, _module != SVModuleNone, XCAM_RETURN_ERROR_PARAM,
        "invalid module, please set module first");

    VideoBufferInfo info;
    info.init (V4L2_PIX_FMT_NV12, get_width (), get_height ());

    SmartPtr<BufferPool> pool;
    if (_module == SVModuleSoft) {
        pool = new SoftVideoBufAllocator (info);
    } else if (_module == SVModuleGLES) {
#if HAVE_GLES
        pool = new GLVideoBufferPool (info);
#endif
    } else if (_module == SVModuleVulkan) {
#if HAVE_VULKAN
        XCAM_ASSERT (_vk_dev.ptr ());
        pool = create_vk_buffer_pool (_vk_dev);
        XCAM_ASSERT (pool.ptr ());
        pool->set_video_info (info);
#endif
    }
    XCAM_ASSERT (pool.ptr ());

    if (!pool->reserve (reserve_count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<Stitcher>
create_stitcher (const SmartPtr<SVStream> &stitch, SVModule module)
{
    SmartPtr<Stitcher> stitcher;

    if (module == SVModuleSoft) {
        stitcher = Stitcher::create_soft_stitcher ();
    } else if (module == SVModuleGLES) {
#if HAVE_GLES
        stitcher = Stitcher::create_gl_stitcher ();
#endif
    } else if (module == SVModuleVulkan) {
#if HAVE_VULKAN
        SmartPtr<VKDevice> dev = stitch->get_vk_device ();
        XCAM_ASSERT (dev.ptr ());
        stitcher = Stitcher::create_vk_stitcher (dev);
#else
        XCAM_UNUSED (stitch);
#endif
    }
    XCAM_ASSERT (stitcher.ptr ());

    return stitcher;
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
add_stream (SVStreams &streams, const char *stream_name, uint32_t width, uint32_t height)
{
    char file_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    combine_name (streams[0]->get_file_name (), stream_name, file_name);

    SmartPtr<SVStream> stream = new SVStream (file_name, width, height);
    XCAM_ASSERT (stream.ptr ());
    streams.push_back (stream);
}

static void
write_in_image (const SVStreams &ins, uint32_t frame_num)
{
#if (XCAM_TEST_STREAM_DEBUG) && (XCAM_TEST_OPENCV)
    char frame_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    std::snprintf (frame_str, XCAM_TEST_MAX_STR_SIZE, "frame:%d", frame_num);

    char img_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    char idx_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    for (uint32_t i = 0; i < ins.size (); ++i) {
        std::snprintf (idx_str, XCAM_TEST_MAX_STR_SIZE, "idx:%d", i);
        std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "orig_fisheye_%d_%d.jpg", frame_num, i);
        ins[i]->debug_write_image (img_name, frame_str, idx_str);
    }
#else
    XCAM_UNUSED (ins);
    XCAM_UNUSED (frame_num);
#endif
}

static void
write_out_image (const SmartPtr<SVStream> &out, uint32_t frame_num)
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

static XCamReturn
create_topview_mapper (
    const SmartPtr<Stitcher> &stitcher, const SmartPtr<SVStream> &stitch,
    const SmartPtr<SVStream> &topview, SVModule module)
{
    BowlModel bowl_model (stitcher->get_bowl_config (), stitch->get_width (), stitch->get_height ());
    BowlModel::PointMap points;

    float length_mm = 0.0f, width_mm = 0.0f;
    bowl_model.get_max_topview_area_mm (length_mm, width_mm);
    XCAM_LOG_INFO ("Max Topview Area (L%.2fmm, W%.2fmm)", length_mm, width_mm);

    bowl_model.get_topview_rect_map (points, topview->get_width (), topview->get_height (), length_mm, width_mm);
    SmartPtr<GeoMapper> mapper;
    if (module == SVModuleSoft) {
        mapper = GeoMapper::create_soft_geo_mapper ();
    } else if (module == SVModuleGLES) {
#if HAVE_GLES
        mapper = GeoMapper::create_gl_geo_mapper ();
#endif
    } else if (module == SVModuleVulkan) {
#if HAVE_VULKAN
        SmartPtr<VKDevice> dev = stitch->get_vk_device ();
        XCAM_ASSERT (dev.ptr ());
        mapper = GeoMapper::create_vk_geo_mapper (dev, "topview-map");
#endif
    }
    XCAM_ASSERT (mapper.ptr ());

    mapper->set_output_size (topview->get_width (), topview->get_height ());
    mapper->set_lookup_table (points.data (), topview->get_width (), topview->get_height ());
    topview->set_mapper (mapper);

    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
remap_topview_buf (const SmartPtr<SVStream> &stitch, const SmartPtr<SVStream> &topview)
{
    const SmartPtr<GeoMapper> mapper = topview->get_mapper();
    XCAM_ASSERT (mapper.ptr ());

    XCamReturn ret = mapper->remap (stitch->get_buf (), topview->get_buf ());
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("remap stitched image to topview failed.");
        return ret;
    }

    return XCAM_RETURN_NO_ERROR;
}

static void
write_image (
    const SVStreams &ins, const SVStreams &outs, bool save_output, bool save_topview)
{
    static uint32_t frame_num = 0;

    write_in_image (ins, frame_num);

    if (save_output)
        write_out_image (outs[IdxStitch], frame_num);

    if (save_topview) {
        remap_topview_buf (outs[IdxStitch], outs[IdxTopView]);
        write_out_image (outs[IdxTopView], frame_num);
    }

    frame_num++;
}

static int
single_frame (
    const SmartPtr<Stitcher> &stitcher,
    const SVStreams &ins, const SVStreams &outs,
    bool save_output, bool save_topview, int loop)
{
    for (uint32_t i = 0; i < ins.size (); ++i) {
        CHECK (ins[i]->rewind (), "rewind buffer from file(%s) failed", ins[i]->get_file_name ());
    }

    VideoBufferList in_buffers;
    for (uint32_t i = 0; i < ins.size (); ++i) {
        XCamReturn ret = ins[i]->read_buf ();
        CHECK_EXP (ret == XCAM_RETURN_NO_ERROR, "read buffer from file(%s) failed.", ins[i]->get_file_name ());

        XCAM_ASSERT (ins[i]->get_buf ().ptr ());
        in_buffers.push_back (ins[i]->get_buf ());
    }

    while (loop--) {
        CHECK (stitcher->stitch_buffers (in_buffers, outs[IdxStitch]->get_buf ()), "stitch buffer failed.");

        if (save_output || save_topview)
            write_image (ins, outs, save_output, save_topview);

        FPS_CALCULATION (surround-view, XCAM_OBJ_DUR_FRAME_NUM);
    }

    return 0;
}

static int
multi_frame (
    const SmartPtr<Stitcher> &stitcher,
    const SVStreams &ins, const SVStreams &outs,
    bool save_output, bool save_topview, int loop)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    VideoBufferList in_buffers;
    while (loop--) {
        for (uint32_t i = 0; i < ins.size (); ++i) {
            CHECK (ins[i]->rewind (), "rewind buffer from file(%s) failed", ins[i]->get_file_name ());
        }

        do {
            in_buffers.clear ();

            for (uint32_t i = 0; i < ins.size (); ++i) {
                ret = ins[i]->read_buf();
                if (ret == XCAM_RETURN_BYPASS)
                    break;
                CHECK (ret, "read buffer from file(%s) failed.", ins[i]->get_file_name ());

                in_buffers.push_back (ins[i]->get_buf ());
            }
            if (ret == XCAM_RETURN_BYPASS)
                break;

            CHECK (
                stitcher->stitch_buffers (in_buffers, outs[IdxStitch]->get_buf ()),
                "stitch buffer failed.");

            if (save_output || save_topview)
                write_image (ins, outs, save_output, save_topview);

            FPS_CALCULATION (surround-view, XCAM_OBJ_DUR_FRAME_NUM);
        } while (true);
    }

    return 0;
}

static int
run_stitcher (
    const SmartPtr<Stitcher> &stitcher,
    const SVStreams &ins, const SVStreams &outs,
    FrameMode frame_mode, bool save_output, bool save_topview, int loop)
{
    CHECK (check_streams<SVStreams> (ins), "invalid input streams");
    CHECK (check_streams<SVStreams> (outs), "invalid output streams");

    int ret = -1;
    if (frame_mode == FrameSingle)
        ret = single_frame (stitcher, ins, outs, save_output, save_topview, loop);
    else if (frame_mode == FrameMulti)
        ret = multi_frame (stitcher, ins, outs, save_output, save_topview, loop);
    else
        XCAM_LOG_ERROR ("invalid frame mode: %d", frame_mode);

    return ret;
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --module MODULE --input input0.nv12 --input input1.nv12 --input input2.nv12 ...\n"
            "\t--module            processing module, selected from: soft, gles, vulkan\n"
            "\t                    read calibration files from exported path $FISHEYE_CONFIG_PATH\n"
            "\t--input             input image(NV12)\n"
            "\t--output            output image(NV12/MP4)\n"
            "\t--in-w              optional, input width, default: 1280\n"
            "\t--in-h              optional, input height, default: 800\n"
            "\t--out-w             optional, output width, default: 1920\n"
            "\t--out-h             optional, output height, default: 640\n"
            "\t--topview-w         optional, output width, default: 1280\n"
            "\t--topview-h         optional, output height, default: 720\n"
            "\t--fisheye-num       optional, the number of fisheye lens, default: 4\n"
            "\t--res-mode          optional, image resolution mode\n"
            "\t                    select from [1080p2cams/1080p4cams/8k3cams], default: 1080p4cams\n"
            "\t--dewarp-mode       optional, fisheye dewarp mode, select from [sphere/bowl], default: bowl\n"
            "\t--scale-mode        optional, scaling mode for geometric mapping,\n"
            "\t                    select from [singleconst/dualconst/dualcurve], default: singleconst\n"
            "\t--fm-mode           optional, feature match mode,\n"
#if HAVE_OPENCV
            "\t                    select from [none/default/cluster/capi], default: none\n"
#else
            "\t                    select from [none], default: none\n"
#endif
            "\t--frame-mode        optional, times of buffer reading, select from [single/multi], default: multi\n"
            "\t--save              optional, save file or not, select from [true/false], default: true\n"
            "\t--save-topview      optional, save top view video, select from [true/false], default: false\n"
            "\t--loop              optional, how many loops need to run, default: 1\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    uint32_t input_width = 1280;
    uint32_t input_height = 800;
    uint32_t output_width = 1920;
    uint32_t output_height = 640;
    uint32_t topview_width = 1280;
    uint32_t topview_height = 720;

    SVStreams ins;
    SVStreams outs;

    uint32_t fisheye_num = 4;
    FrameMode frame_mode = FrameMulti;
    SVModule module = SVModuleNone;
    GeoMapScaleMode scale_mode = ScaleSingleConst;
    FeatureMatchMode fm_mode = FMNone;
    StitchResMode res_mode = StitchRes1080P4Cams;
    FisheyeDewarpMode dewarp_mode = DewarpBowl;

    int loop = 1;
    bool save_output = true;
    bool save_topview = false;

    const struct option long_opts[] = {
        {"module", required_argument, NULL, 'm'},
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"topview-w", required_argument, NULL, 'P'},
        {"topview-h", required_argument, NULL, 'V'},
        {"fisheye-num", required_argument, NULL, 'N'},
        {"res-mode", required_argument, NULL, 'R'},
        {"dewarp-mode", required_argument, NULL, 'd'},
        {"scale-mode", required_argument, NULL, 'S'},
        {"fm-mode", required_argument, NULL, 'F'},
        {"frame-mode", required_argument, NULL, 'f'},
        {"save", required_argument, NULL, 's'},
        {"save-topview", required_argument, NULL, 't'},
        {"loop", required_argument, NULL, 'L'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'm':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "soft"))
                module = SVModuleSoft;
            else if (!strcasecmp (optarg, "gles")) {
                module = SVModuleGLES;
            } else if (!strcasecmp (optarg, "vulkan")) {
                module = SVModuleVulkan;
            } else {
                XCAM_LOG_ERROR ("unknown module:%s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'i':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SVStream, ins, optarg);
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SVStream, outs, optarg);
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
        case 'P':
            topview_width = atoi(optarg);
            break;
        case 'V':
            topview_height = atoi(optarg);
            break;
        case 'N':
            fisheye_num = atoi(optarg);
            if (fisheye_num > XCAM_STITCH_FISHEYE_MAX_NUM) {
                XCAM_LOG_ERROR ("fisheye number should not be greater than %d\n", XCAM_STITCH_FISHEYE_MAX_NUM);
                return -1;
            }
            break;
        case 'R':
            if (!strcasecmp (optarg, "1080p2cams"))
                res_mode = StitchRes1080P2Cams;
            else if (!strcasecmp (optarg, "1080p4cams"))
                res_mode = StitchRes1080P4Cams;
            else if (!strcasecmp (optarg, "8k3cams"))
                res_mode = StitchRes8K3Cams;
            else {
                XCAM_LOG_ERROR ("incorrect resolution mode");
                return -1;
            }
            break;
        case 'd':
            if (!strcasecmp (optarg, "sphere"))
                dewarp_mode = DewarpSphere;
            else if(!strcasecmp (optarg, "bowl"))
                dewarp_mode = DewarpBowl;
            else {
                XCAM_LOG_ERROR ("incorrect fisheye dewarp mode");
                return -1;
            }
            break;
        case 'S':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "singleconst"))
                scale_mode = ScaleSingleConst;
            else if (!strcasecmp (optarg, "dualconst"))
                scale_mode = ScaleDualConst;
            else if (!strcasecmp (optarg, "dualcurve"))
                scale_mode = ScaleDualCurve;
            else {
                XCAM_LOG_ERROR ("GeoMapScaleMode unknown mode: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'F':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "none"))
                fm_mode = FMNone;
#if HAVE_OPENCV
            else if (!strcasecmp (optarg, "default"))
                fm_mode = FMDefault;
            else if (!strcasecmp (optarg, "cluster"))
                fm_mode = FMCluster;
            else if (!strcasecmp (optarg, "capi"))
                fm_mode = FMCapi;
#endif
            else {
                XCAM_LOG_ERROR ("surround view unsupported feature match mode: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'f':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "single"))
                frame_mode = FrameSingle;
            else if (!strcasecmp (optarg, "multi"))
                frame_mode = FrameMulti;
            else {
                XCAM_LOG_ERROR ("FrameMode unknown mode: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 's':
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 't':
            save_topview = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'L':
            loop = atoi(optarg);
            break;
        case 'e':
            usage (argv[0]);
            return 0;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value: %c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknown option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    if ((ins.size () != 1) && (ins.size () != fisheye_num)) {
        XCAM_LOG_ERROR (
            "multiple-input mode: conflicting input number(%d) and fisheye number(%d)", ins.size (), fisheye_num);
        return -1;
    }

    for (uint32_t i = 0; i < ins.size (); ++i) {
        CHECK_EXP (ins[i].ptr (), "input stream is NULL, index:%d", i);
        CHECK_EXP (strlen (ins[i]->get_file_name ()), "input file name was not set, index:%d", i);
    }

    CHECK_EXP (outs.size () == 1 && outs[IdxStitch].ptr (), "surrond view needs 1 output stream");
    CHECK_EXP (strlen (outs[IdxStitch]->get_file_name ()), "output file name was not set");

    for (uint32_t i = 0; i < ins.size (); ++i) {
        printf ("input%d file:\t\t%s\n", i, ins[i]->get_file_name ());
    }
    printf ("output file:\t\t%s\n", outs[IdxStitch]->get_file_name ());
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("topview width:\t\t%d\n", topview_width);
    printf ("topview height:\t\t%d\n", topview_height);
    printf ("fisheye number:\t\t%d\n", fisheye_num);
    printf ("resolution mode:\t%s\n", res_mode == StitchRes1080P2Cams ? "1080p2cams" :
            (res_mode == StitchRes1080P4Cams ? "1080p4cams" : "8k3cams"));
    printf ("dewarp mode: \t\t%s\n", dewarp_mode == DewarpSphere ? "sphere" : "bowl");
    printf ("scaling mode:\t\t%s\n", (scale_mode == ScaleSingleConst) ? "singleconst" :
            ((scale_mode == ScaleDualConst) ? "dualconst" : "dualcurve"));
    printf ("feature match:\t\t%s\n", (fm_mode == FMNone) ? "none" :
            ((fm_mode == FMDefault ) ? "default" : ((fm_mode == FMCluster) ? "cluster" : "capi")));
    printf ("frame mode:\t\t%s\n", (frame_mode == FrameSingle) ? "singleframe" : "multiframe");
    printf ("save output:\t\t%s\n", save_output ? "true" : "false");
    printf ("save topview:\t\t%s\n", save_topview ? "true" : "false");
    printf ("loop count:\t\t%d\n", loop);

#if HAVE_GLES
    SmartPtr<EGLBase> egl;
    if (module == SVModuleGLES) {
        if (scale_mode == ScaleDualCurve) {
            XCAM_LOG_ERROR ("GLES module does not support dualcurve scale mode currently");
            return -1;
        }

        egl = new EGLBase ();
        XCAM_ASSERT (egl.ptr ());
        XCAM_FAIL_RETURN (ERROR, egl->init (), -1, "init EGL failed");
    }
#else
    if (module == SVModuleGLES) {
        XCAM_LOG_ERROR ("GLES module is unsupported");
        return -1;
    }
#endif

    if (module == SVModuleVulkan) {
#if HAVE_VULKAN
        if (scale_mode != ScaleSingleConst) {
            XCAM_LOG_ERROR ("vulkan module only support singleconst scale mode currently");
            return -1;
        }

        SmartPtr<VKDevice> vk_dev = VKDevice::default_device ();
        for (uint32_t i = 0; i < ins.size (); ++i) {
            ins[i]->set_vk_device (vk_dev);
        }
        XCAM_ASSERT (outs[IdxStitch].ptr ());
        outs[IdxStitch]->set_vk_device (vk_dev);
#else
        XCAM_LOG_ERROR ("vulkan module is unsupported");
        return -1;
#endif
    }

    for (uint32_t i = 0; i < ins.size (); ++i) {
        ins[i]->set_module (module);
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

    SmartPtr<Stitcher> stitcher = create_stitcher (outs[IdxStitch], module);
    XCAM_ASSERT (stitcher.ptr ());

    stitcher->set_camera_num (fisheye_num);
    stitcher->set_output_size (output_width, output_height);
    stitcher->set_res_mode (res_mode);
    stitcher->set_dewarp_mode (dewarp_mode);
    stitcher->set_scale_mode (scale_mode);
    stitcher->set_fm_mode (fm_mode);

    if (dewarp_mode == DewarpSphere) {
        if (res_mode == StitchRes1080P2Cams) {
            float viewpoints_range[] = {202.8f, 202.8f};
            stitcher->set_viewpoints_range (viewpoints_range);
        } else if (res_mode == StitchRes8K3Cams) {
            float viewpoints_range[] = {144.0f, 144.0f, 144.0f};
            stitcher->set_viewpoints_range (viewpoints_range);
        }
    } else {
        float viewpoints_range[] = {64.0f, 160.0f, 64.0f, 160.0f};
        stitcher->set_viewpoints_range (viewpoints_range);

        stitcher->set_instrinsic_names (instrinsic_names);
        stitcher->set_exstrinsic_names (exstrinsic_names);

        BowlDataConfig bowl;
        bowl.wall_height = 1800.0f;
        bowl.ground_length = 3000.0f;
        bowl.angle_start = 0.0f;
        bowl.angle_end = 360.0f;
        stitcher->set_bowl_config (bowl);
    }

    if (save_topview) {
        add_stream (outs, "topview", topview_width, topview_height);
        XCAM_ASSERT (outs.size () >= IdxCount);

        CHECK (outs[IdxTopView]->estimate_file_format (),
            "%s: estimate file format failed", outs[IdxTopView]->get_file_name ());
        CHECK (outs[IdxTopView]->open_writer ("wb"), "open output file(%s) failed", outs[IdxTopView]->get_file_name ());

        create_topview_mapper (stitcher, outs[IdxStitch], outs[IdxTopView], module);
    }

    CHECK_EXP (
        run_stitcher (stitcher, ins, outs, frame_mode, save_output, save_topview, loop) == 0,
        "run stitcher failed");

    return 0;
}
