#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

m4_define([xcam_major_version], [1])
m4_define([xcam_minor_version], [5])
m4_define([xcam_micro_version], [0])
m4_define([xcam_version], [xcam_major_version.xcam_minor_version.xcam_micro_version])

AC_PREREQ([2.60])
AC_CONFIG_MACRO_DIR([m4])
AC_INIT([libxcam], [xcam_version], [wei.zong@intel.com], [libxcam])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([foreign subdir-objects])

# xcam version
XCAM_MAJOR_VERSION=xcam_major_version
XCAM_MINOR_VERSION=xcam_minor_version
XCAM_MICRO_VERSION=xcam_micro_version
XCAM_VERSION=xcam_version
XCAM_VERSION_HEX="0x$XCAM_MAJOR_VERSION$XCAM_MINOR_VERSION$XCAM_MICRO_VERSION"
AC_DEFINE_UNQUOTED(XCAM_VERSION, $XCAM_VERSION_HEX,
    [define libxcam version])
AC_SUBST(XCAM_VERSION)

XCAM_LT_VERSION="xcam_major_version:xcam_minor_version:xcam_micro_version"
XCAM_LT_LDFLAGS="-version-number $XCAM_LT_VERSION"
AC_SUBST(XCAM_LT_VERSION)
AC_SUBST(XCAM_LT_LDFLAGS)

# xcam required OpenCV version [XCAM_REQUIRE_CV_MIN, XCAM_REQUIRE_CV_MAX)
XCAM_REQUIRE_CV_MIN=3.0.0
XCAM_REQUIRE_CV_MAX=5.0.0

# xcam required OpenSceneGraph version
XCAM_REQUIRE_OSG_MIN=3.3.2

# xcam required OpenVino version
XCAM_REQUIRE_DNN_MIN=2023.0.0

# xcam gstreamer version
GST_API_VERSION=1.0
GST_VERSION_MIN=1.2.3

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
PKG_PROG_PKG_CONFIG
LT_INIT

# set XCAM_CFLAGS and XCAM_CXXFLAGS
XCAM_CFLAGS=" -fPIC -Wall -fstack-protector"
XCAM_IF($enable_debug, yes, XCAM_CFLAGS="$XCAM_CFLAGS -g -DDEBUG")
XCAM_CXXFLAGS="$XCAM_CFLAGS -std=c++0x"
AC_SUBST(XCAM_CFLAGS)
AC_SUBST(XCAM_CXXFLAGS)

XCAM_ARG_ENABLE(debug, --enable-debug, enable_debug, no, enable debug)
XCAM_ARG_ENABLE(profiling, --enable-profiling, enable_profiling, no, enable profiling)
XCAM_ARG_ENABLE(drm, --enable-drm, enable_drm, no, enable drm buffer)
XCAM_ARG_ENABLE(aiq, --enable-aiq, enable_aiq, no, enable Aiq 3A algorithm)
XCAM_ARG_ENABLE(gst, --enable-gst, enable_gst, no, enable gstreamer plugin)
XCAM_ARG_ENABLE(libcl, --enable-libcl, enable_libcl, yes, enable libcl image processor)
XCAM_ARG_ENABLE(gles, --enable-gles, enable_gles, no, enable gles)
XCAM_ARG_ENABLE(vulkan, --enable-vulkan, enable_vulkan, no, enable vulkan)
XCAM_ARG_ENABLE(avx512, --enable-avx512, enable_avx512, no, enable avx512)
XCAM_ARG_ENABLE(opencv, --enable-opencv, enable_opencv, no, enable opencv library)
XCAM_ARG_ENABLE(capi, --enable-capi, enable_capi, no, enable libxcam-capi library)
XCAM_ARG_ENABLE(render, --enable-render, enable_render, no, enable texture render with OpenSceneGraph library)
XCAM_ARG_ENABLE(dnn, --enable-dnn, enable_dnn, no, enable dnn with OpenVino library)
XCAM_ARG_ENABLE(docs, --enable-docs, enable_docs, no, build Doxygen documentation)
XCAM_ARG_ENABLE(3alib, --enable-3alib, enable_3alib, no, enable 3A library)
XCAM_ARG_ENABLE(smartlib, --enable-smartlib, enable_smartlib, no, enable smart analysis library)
XCAM_ARG_ENABLE(json, --enable-json, enable_json, no, enable camera calibration json parser)

XCAM_CHECK_MODULE($enable_drm, libdrm, LIBDRM, HAVE_LIBDRM=1, HAVE_LIBDRM=0)
XCAM_CHECK_MODULE($enable_libcl, OpenCL, LIBCL, HAVE_LIBCL=1, HAVE_LIBCL=0)
XCAM_CHECK_MODULE($enable_gles, gl, LIBGL, HAVE_GLES=1, HAVE_GLES=0)
XCAM_CHECK_MODULE($enable_gles, gbm, LIBGBM, HAVE_GBM=1, HAVE_GBM=0)
XCAM_CHECK_MODULE($enable_vulkan, vulkan, LIBVULKAN, HAVE_VULKAN=1, HAVE_VULKAN=0)

XCAM_CHECK_GAWK($HAVE_LIBCL, $HAVE_GLES)
XCAM_CHECK_DOXYGEN($enable_docs, [], enable_docs="no")
XCAM_CHECK_AVX512($enable_avx512, ENABLE_AVX512=1, ENABLE_AVX512=0)
XCAM_CHECK_OSG($enable_render, $XCAM_REQUIRE_OSG_MIN, ENABLE_RENDER=1, ENABLE_RENDER=0)
XCAM_CHECK_DNN($enable_dnn, $XCAM_REQUIRE_DNN_MIN, $OPENVINO_EXTRAS_PATH, ENABLE_DNN=1, ENABLE_DNN=0)
XCAM_CHECK_OPENCV($enable_opencv, $XCAM_REQUIRE_CV_MIN, $XCAM_REQUIRE_CV_MAX, HAVE_OPENCV=1, HAVE_OPENCV=0, OPENCV_VERSION3=1, OPENCV_VERSION3=0)
XCAM_CHECK_OCV_VIDEOSTAB($HAVE_OPENCV, ENABLE_DVS=1, ENABLE_DVS=0)
XCAM_CHECK_DVS_OCL($HAVE_OPENCV, ENABLE_DVS_CL_PATH=1, ENABLE_DVS_CL_PATH=0)
XCAM_CHECK_GST($enable_gst, $GST_API_VERSION, $GST_VERSION_MIN, ENABLE_GST=1, ENABLE_GST=0)
XCAM_CHECK_AIQ($enable_aiq, ENABLE_IA_AIQ=1, ENABLE_IA_AIQ=0, USE_LOCAL_AIQ=1, USE_LOCAL_AIQ=0)
XCAM_CHECK_LOCAL_ATOMISP($enable_aiq, USE_LOCAL_ATOMISP=1, USE_LOCAL_ATOMISP=0)
XCAM_CHECK_JSON($enable_json, HAVE_JSON=1, HAVE_JSON=0)

XCAM_IF($enable_capi, yes, ENABLE_CAPI=1, ENABLE_CAPI=0)
XCAM_IF($enable_profiling, yes, ENABLE_PROFILING=1, ENABLE_PROFILING=0)
XCAM_IF($enable_3alib, yes, ENABLE_3ALIB=1, ENABLE_3ALIB=0)
XCAM_IF($enable_smartlib, yes, ENABLE_SMART_LIB=1, ENABLE_SMART_LIB=0)

XCAM_PKG_EXPORT_LIBS=" -lxcam_core -lxcam_soft"
XCAM_CHECK_PKG_CONFIG($HAVE_LIBCL, 1, XCAM_PKG_EXPORT_LIBS+=" -lxcam_ocl")
XCAM_CHECK_PKG_CONFIG($HAVE_GLES, 1, XCAM_PKG_EXPORT_LIBS+=" -lxcam_gles")
XCAM_CHECK_PKG_CONFIG($HAVE_VULKAN, 1, XCAM_PKG_EXPORT_LIBS+=" -lxcam_vulkan")
XCAM_CHECK_PKG_CONFIG($ENABLE_RENDER, 1, XCAM_PKG_EXPORT_LIBS+=" -lxcam_render")
XCAM_CHECK_PKG_CONFIG($ENABLE_DNN, 1, XCAM_PKG_EXPORT_LIBS+=" -lxcam_dnn")
XCAM_CHECK_PKG_CONFIG($ENABLE_CAPI, 1, XCAM_PKG_EXPORT_LIBS+=" -lxcam_capi")

# define macor in config.h
AC_DEFINE_UNQUOTED(XCAM_PKG_EXPORT_LIBS, "$XCAM_PKG_EXPORT_LIBS", [define libxcam pkg libs])
AC_SUBST(XCAM_PKG_EXPORT_LIBS)

XCAM_DEFINE_MACOR(ENABLE_PROFILING, $ENABLE_PROFILING, enable profiling)
XCAM_DEFINE_MACOR(HAVE_LIBDRM, $HAVE_LIBDRM, have libdrm)
XCAM_DEFINE_MACOR(HAVE_LIBCL, $HAVE_LIBCL, have libcl)
XCAM_DEFINE_MACOR(HAVE_GLES, $HAVE_GLES, have gles)
XCAM_DEFINE_MACOR(HAVE_GBM, $HAVE_GBM, have gbm)
XCAM_DEFINE_MACOR(HAVE_VULKAN, $HAVE_VULKAN, have vulkan)
XCAM_DEFINE_MACOR(ENABLE_AVX512, $ENABLE_AVX512, have avx instruction)
XCAM_DEFINE_MACOR(ENABLE_RENDER, $ENABLE_RENDER, enable texture render)
XCAM_DEFINE_MACOR(ENABLE_DNN, $ENABLE_DNN, have dnn)
XCAM_DEFINE_MACOR(HAVE_OPENCV, $HAVE_OPENCV, have opencv)
XCAM_DEFINE_MACOR(OPENCV_VERSION3, $OPENCV_VERSION3, opencv version3)
XCAM_DEFINE_MACOR(ENABLE_DVS, $ENABLE_DVS, have dvs)
XCAM_DEFINE_MACOR(ENABLE_DVS_CL_PATH, $ENABLE_DVS_CL_PATH, enable dvs cl path)
XCAM_DEFINE_MACOR(ENABLE_CAPI, $ENABLE_CAPI, enable capi)
XCAM_DEFINE_MACOR(HAVE_IA_AIQ, $ENABLE_IA_AIQ, have aiq binary)
XCAM_DEFINE_MACOR(HAVE_JSON, $HAVE_JSON, have json)

XCAM_CONDITIONAL(DEBUG, $enable_debug, yes)
XCAM_CONDITIONAL(ENABLE_DOCS, $enable_docs, yes)
XCAM_CONDITIONAL(ENABLE_3ALIB, $ENABLE_3ALIB, 1)
XCAM_CONDITIONAL(ENABLE_SMART_LIB, $ENABLE_SMART_LIB, 1)
XCAM_CONDITIONAL(ENABLE_GST, $ENABLE_GST, 1)
XCAM_CONDITIONAL(USE_LOCAL_ATOMISP, $USE_LOCAL_ATOMISP, 1)
XCAM_CONDITIONAL(ENABLE_IA_AIQ, $ENABLE_IA_AIQ, 1)
XCAM_CONDITIONAL(USE_LOCAL_AIQ, $USE_LOCAL_AIQ, 1)

AC_CONFIG_FILES([Makefile
                 shaders/Makefile
                 shaders/clx/Makefile
                 shaders/glslx/Makefile
                 xcore/Makefile
                 modules/Makefile
                 modules/ocv/Makefile
                 modules/soft/Makefile
                 modules/isp/Makefile
                 modules/ocl/Makefile
                 modules/gles/Makefile
                 modules/vulkan/Makefile
                 modules/render/Makefile
                 modules/dnn/Makefile
                 modules/dnn/inference/Makefile
                 wrapper/Makefile
                 wrapper/gstreamer/Makefile
                 wrapper/gstreamer/interface/Makefile
                 plugins/Makefile
                 plugins/3a/hybrid/Makefile
                 plugins/3a/aiq/Makefile
                 plugins/3a/Makefile
                 plugins/smart/Makefile
                 plugins/smart/dvs/Makefile
                 plugins/smart/dvs/libdvs/Makefile
                 plugins/smart/sample/Makefile
                 capi/Makefile
                 tests/Makefile
                 pkgconfig/Makefile
                 pkgconfig/libxcam.pc
                ])

AC_OUTPUT

XCAM_IF($HAVE_LIBDRM, 1, have_drm="yes", have_drm="no")
XCAM_IF($USE_LOCAL_AIQ, 1, use_local_aiq="yes", use_local_aiq="no")
XCAM_IF($USE_LOCAL_ATOMISP, 1, use_local_atomisp="yes", use_local_atomisp="no")
XCAM_IF($HAVE_LIBCL, 1, have_libcl="yes", have_libcl="no")
XCAM_IF($HAVE_GLES, 1, have_gles="yes", have_gles="no")
XCAM_IF($HAVE_VULKAN, 1, have_vulkan="yes", have_vulkan="no")
XCAM_IF($ENABLE_AVX512, 1, enable_avx512="yes", enable_avx512="no")
XCAM_IF($HAVE_OPENCV, 1, have_opencv="yes", have_opencv="no")
XCAM_IF($ENABLE_RENDER, 1, enable_render="yes", enable_render="no")
XCAM_IF($ENABLE_DNN, 1, enable_dnn="yes", enable_dnn="no")
XCAM_IF($ENABLE_DVS, 1, enable_dvs="yes", enable_dvs="no")
XCAM_IF($ENABLE_CAPI, 1, enable_capi="yes", enable_capi="no")
XCAM_IF($HAVE_JSON, 1, have_json="yes", have_json="no")


echo "
     libxcam configuration summary
     version                    : $XCAM_VERSION
     enable debug               : $enable_debug
     enable profiling           : $enable_profiling
     enable drm lib             : $have_drm
     build GStreamer plugin     : $enable_gst
     build aiq analyzer         : $enable_aiq
     use local aiq              : $use_local_aiq
     use local atomisp          : $use_local_atomisp
     enable OpenCV              : $have_opencv
     enable OpenCL              : $have_libcl
     enable GLES                : $have_gles
     enable Vulkan              : $have_vulkan
     enable AVX512 instruction  : $enable_avx512
     enable OSG render          : $enable_render
     enable DNN                 : $enable_dnn
     enable 3a lib              : $enable_3alib
     enable smart analysis lib  : $enable_smartlib
     enable dvs                 : $enable_dvs
     enable libxcam-capi lib    : $enable_capi
     enable json parser         : $have_json
"
