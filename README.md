# PROJECT NOT UNDER ACTIVE MANAGEMENT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
## libXCam

Copyright (C) 2014-2023 Intel Corporation

libxcam core source code under the terms of Apache License, Version 2.0

#### Description:
libXCam is a project for extended camera features and focus on image
quality improvement and video analysis. There are lots features supported
in image pre-processing, image post-processing and smart analysis. This
library makes GPU/CPU/ISP working together to improve image quality.
OpenCL is used to improve performance in different platforms.

#### Features:
  * Image processing features
    - Advanced features
      - 360-degree video stitching
        - support ultra-high resolution (8K) video stitching.
        - support processing NV12 & YUV420 pixel format.
        - support 2/3/4 fish-eye cameras (FoV >180 degree) video stitching.
        - stitching image adopts equirectangular projection (ERP).
        - algorithms are optimized by AVX512 instruction, GLES and Vulkan.
        - stitching quality tuning supports OpenCV fisheye camera calibration parameters.
      - Automotive surround view (360-degree) stitching (OpenCL/CPU/GLES)
         - Support bowl view 3D model stitching by 4 input videos.
         - Enable geometry remap for WFoV camera calibration(intrinsic and extrinsic data).
         - Quality and performance improved (OpenCL/CPU/GLES).
         - CPU version upstreamed into AOSP for automotive surround view.
         - Enable Vulkan to improve performance.
      - DNN inference framework
        - Support pedestrian and vehicle detection.
      - Digital Video Stabilization
        - OpenCV feature-matched based video stabilization.
        - gyroscope 3-DoF (orientation) based video stabilization.
      - Blender: multi-band blender (OpenCL/CPU/GLES)
      - Noise reduction (OpenCL)
        - adaptive NR based on wavelet-haar and Bayersian shrinkage.
        - 3D-NR with inter-block and intra-block reference.
        - wavelet-hat NR (obsolete).
      - Wide dynamic range (WDR) (OpenCL)
        - histogram adjustment tone-mapping.
        - gaussian-based tone-mapping (obsolete).
      - Fog removal: retinex and dark channel prior algorithm (OpenCL)
        - dark channel prior algorithm based defog.
        - multi-scale retinex based defog (obsolete).
    - Basic pipeline from bayer to YUV/RGB format (OpenCL / AtomISP)
      - Gamma correction, MACC, color space, demosaicing, simple bilateral
        noise reduction, edge enhancement and temporal noise reduction.
    - 3A features
      - Auto whitebalance, auto exposure, auto focus, black level correction,
        color correction, 3a-statistics calculation.
  * Support 3rd party 3A lib which can be loaded dynamically
       - hybrid 3a plugin.
  * Support 3a analysis tuning framework for different features
  * Support smart analysis framework
       - Face detection interface/plugin.
  * Enable gstreamer plugin
       - xcamsrc, capture from usb/isp camera, process 3a/basic/advanced features.
       - xcamfilter, improve image quality by advanced features and smart analysis.

#### Prerequisite:
  * install gcc/g++, automake, autoconf, libtool, gawk, pkg-config
  * Linux kernel > 3.10
  * install ocl-icd-dev, ocl-icd-opencl-dev
  * If --enable-gst, need install libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev
  * If --enable-aiq, need get ia_imaging lib which we don't support
  * If --enable-libcl, need compile or install [OpenCL](https://github.com/intel/compute-runtime) driver
  * If --enable-opencv, suggest [OpenCV](http://opencv.org) versions [3.0.0 - 4.0.0) (or: [OpenCV Wiki](https://github.com/opencv/opencv/wiki))
  * If --enable-render, need compile [OpenSceneGraph](https://github.com/openscenegraph/OpenSceneGraph) library with configure option "-DOSG_WINDOWING_SYSTEM=X11"
  * If --enable-gles, need to install [Mesa3D](https://www.mesa3d.org) library
  * If --enable-vulkan, need to install [Mesa3D](https://www.mesa3d.org) library
  * If --enable-avx512, need to install [GCC version > 4.9](https://gcc.gnu.org/gcc-4.9/changes.html) ([ CPUs With AVX-512 required](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX-512))
  * If --enable-dnn, need to compile [OpenVino](https://github.com/openvinotoolkit/openvino)
  * If --enable-json, need to install [json.hpp](https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp)

#### Building and installing:
  * Environment variable settings<BR>
    For different --prefix options, the environment variables may be different. Please set the environment variable according to the actual situation.<BR>
    --prefix=/usr/local:

        export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
        export GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0:$GST_PLUGIN_PATH
        source OPENVINO_INSTALLDIR/setupvars.sh

    --prefix=/usr:

        export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH
        export GST_PLUGIN_PATH=/usr/lib/gstreamer-1.0:$GST_PLUGIN_PATH
        source OPENVINO_INSTALLDIR/setupvars.sh

  * $ ./autogen.sh [options]

        --prefix=PREFIX         install architecture-independent files in PREFIX [default=/usr/local]
        --enable-debug          enable debug, [default=no]
        --enable-profiling      enable profiling, [default=no]
        --enable-drm            enable drm buffer, [default=no]
        --enable-aiq            enable Aiq 3A algorithm build, [default=no]
        --enable-gst            enable gstreamer plugin build, [default=no]
        --enable-libcl          enable libcl image processor, [default=yes]
        --enable-opencv         enable opencv library, [default=no]
        --enable-capi           enable libxcam-capi library, [default=no]
        --enable-docs           build Doxygen documentation [default=no]
        --enable-3alib          enable 3A lib build, [default=no]
        --enable-smartlib       enable smart analysis lib build, [default=no]
        --enable-gles           enable gles, [default=no]
        --enable-vulkan         enable vulkan, [default=no]
        --enable-avx512         enable AVX-512 instruction, [default=no]
        --enable-render         enable 3D texture render, [default=no]
        --enable-dnn            enable dnn inference, [default=no]
        --enable-json           enable json parser, [default=no]

    For example:

        $ ./autogen.sh --prefix=/usr --enable-gst --enable-libcl --enable-opencv \
          --enable-smartlib --enable-profiling --enable-gles --enable-render --enable-dnn

  * $ make
  * $ sudo make install

#### Testing:
  * For detailed test cases, please go to [tests](https://github.com/intel/libxcam/wiki/Tests) page

#### Reporting Bugs:
  * Bugs or suggestions can be reported on the github [issues](https://github.com/intel/libxcam/issues) page
  * Security issues, please send email to wei.zong@intel.com directly

#### Mailing list
  * To post a message to all the list members, please send email to libxcam@lists.01.org
  * To register libxcam public maillist, please go to [registration](https://lists.01.org/mailman/listinfo/libxcam) page

#### Maintainer:
  * Wind Yuan <feng.yuan@intel.com>
  * Wei Zong <wei.zong@intel.com>

#### Contributors: (orders by first name)
  * Ali Mansouri <ali.m.t1992@gmail.com>
  * Andrey Parfenov <a1994ndrey@gmail.com>
  * Fei Wang <feix.w.wang@intel.com>
  * Jia Meng <jia.meng@intel.com>
  * John Ye <john.ye@intel.com>
  * Juan Zhao <juan.j.zhao@intel.com>
  * Junkai Wu <junkai.wu@intel.com>
  * Sameer Kibey <sameer.kibey@intel.com>
  * Shincy Tu <shincy.tu@intel.com>
  * Wei Zong <wei.zong@intel.com>
  * Yan Zhang <yan.y.zhang@intel.com>
  * Yao Wang <yao.y.wang@intel.com>
  * Yinhang Liu <yinhangx.liu@intel.com>
