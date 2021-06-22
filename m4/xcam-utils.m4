
# XCAM_IF([$1:variable], [$2:value], [$3:if-true], [$4:if-false])
AC_DEFUN([XCAM_IF],
[
    AS_IF([test "x$1" = "x$2"], [$3], [$4])
])

# XCAM_ARG_ENABLE([$1:flag], [$2:string], [$3:variable], [$4:default], [$5:help-info])
AC_DEFUN([XCAM_ARG_ENABLE],
[
    AC_ARG_ENABLE([$1],
        AS_HELP_STRING([$2], [$5, @<:@default=$4@:>@]),
        [], [$3=$4])
])

# XCAM_CONDITIONAL([$1:macor], [$2:variable], [$3:value)
AC_DEFUN([XCAM_CONDITIONAL],
[
    AM_CONDITIONAL($1, [test "x$2" = "x$3"])
])

# XCAM_DEFINE_MACOR([$1:macor], [$2:variable], [$3:help-info)
AC_DEFUN([XCAM_DEFINE_MACOR],
[
    AC_DEFINE_UNQUOTED($1, $2, $3)
    AM_CONDITIONAL($1, [test "x$2" = "x1"])
])

# XCAM_CHECK_GAWK([$1:ocl-value], [$2:gles-value])
AC_DEFUN([XCAM_CHECK_GAWK],
[
    AS_IF([test "x$1" = "x1" || test "x$2" = "x1"],
        [
            AC_CHECK_PROGS(GAWK, gawk, no)
            AS_IF([test "x$GAWK" = "xno"], [AC_MSG_ERROR(gawk not found)])
        ])
])

# XCAM_CHECK_MODULE([$1:value], [$2:module], [$3:prefix], [$4:if-found], [$5:if-not-found])
AC_DEFUN([XCAM_CHECK_MODULE],
[
    AS_IF([test "x$1" = "xyes"],
        [PKG_CHECK_MODULES($3, $2, $4, $5)],
        [$5])
])

# XCAM_CHECK_DOXYGEN([$1:value], [$2:if-found], [$3:if-not-found])
AC_DEFUN([XCAM_CHECK_DOXYGEN],
[
    AS_IF([test "x$1" = "xyes"],
        [
            AC_CHECK_TOOL([DOXYGEN], [doxygen], [no])
            AS_IF([test "x$DOXYGEN" = "xno"], $3, [$2])
        ],
        [$3])
])

# XCAM_CHECK_OPENCV([$1:value], [$2:ocv-min], [$3:ocv-max], [$4:if-found], [$5:if-not-found])
AC_DEFUN([XCAM_CHECK_OPENCV],
[
    AS_IF([test "x$1" = "xyes"],
        [
            ocv_version=`opencv_version`
            AS_IF([test -z $ocv_version], [ocv_version=`$PKG_CONFIG --modversion opencv`])
            AC_MSG_NOTICE(OpenCV version: $ocv_version)

            ocv_module=opencv
            ocv_major_version=`echo $ocv_version | cut -d '.' -f 1`
            AS_IF([test $ocv_major_version -ge 4], [ocv_module=opencv$ocv_major_version])

            PKG_CHECK_MODULES([OPENCV], [$ocv_module >= $2 $ocv_module < $3], [$4],
                [AC_MSG_ERROR(OpenCV required version: >= $2 && < $3)])
        ],
        [$5])
])

# XCAM_CHECK_OCV_VIDEOSTAB([$1:value], [$2:if-found], [$3:if-not-found])
AC_DEFUN([XCAM_CHECK_OCV_VIDEOSTAB],
[
    AS_IF([test "x$1" = "x1"],
        [
            AC_LANG(C++)
            saved_CPPFLAGS="$CPPFLAGS"
            saved_LIBS="$LIBS"

            CPPFLAGS="$CPPFLAGS $OPENCV_CFLAGS"
            LIBS="$LIBS $OPENCV_LIBS"
            AC_CHECK_HEADER([opencv2/videostab.hpp], [$2], [$3])
            CPPFLAGS="$saved_CPPFLAGS"
            LIBS="$saved_LIBS"
        ],
        [$3])
])

# XCAM_CHECK_DVS_OCL([$1:value], [$2:if-found], [$3:if-not-found])
AC_DEFUN([XCAM_CHECK_DVS_OCL],
[
    AS_IF([test "x$1" = "x1"],
        [
            AC_LANG(C++)
            saved_CPPFLAGS="$CPPFLAGS"
            saved_LIBS="$LIBS"

            CPPFLAGS="$CPPFLAGS $OPENCV_CFLAGS"
            LIBS="$LIBS $OPENCV_LIBS"
            AC_COMPILE_IFELSE(
                [
                    AC_LANG_PROGRAM([[
                        #include <opencv2/core.hpp>
                        #include <opencv2/opencv.hpp>
                    ]], [[
                        cv::UMat frame0;
                        cv::UMat frame1;
                        cv::videostab::MotionEstimatorRansacL2 est;
                        cv::videostab::KeypointBasedMotionEstimator kpest(&est);
                        kpest.estimate(frame0, frame1);
                    ]])
                ],
                [$2], [$3]
            )
            CPPFLAGS="$saved_CPPFLAGS"
            LIBS="$saved_LIBS"
        ],
        [$3])
])

# XCAM_CHECK_AVX512([$1:value], [$2:if-found], [$3:if-not-found])
AC_DEFUN([XCAM_CHECK_AVX512],
[
    AS_IF([test "x$1" = "xyes"],
        [
            count=`grep -c avx512 /proc/cpuinfo`
            AS_IF([test $count -gt 0], [], [AC_MSG_WARN(the processor does not support AVX512 instructions)])
            [$2]
        ],
        [$3])
])

# XCAM_CHECK_JSON([$1:value], [$2:if-found], [$3:if-not-found])
AC_DEFUN([XCAM_CHECK_JSON],
[
    AS_IF([test "x$1" = "xyes"],
        [
            #wget -c https://github.com/nlohmann/json/releases/download/v3.7.3/json.hpp

            AC_LANG(C++)
            AC_CHECK_HEADER([json.hpp], [$2], [$3])
        ],
        [$3])
])

# XCAM_CHECK_OSG([$1:value], [$2:osg-min], [$3:if-found], [$4:if-not-found])
AC_DEFUN([XCAM_CHECK_OSG],
[
    AS_IF([test "x$1" = "xyes"],
        [
            PKG_CHECK_MODULES([LIBOSG], [openscenegraph-osg >= $2], [have_osg=1], [have_osg=0])
            AS_IF([test $have_osg = 1],
                [
                    osg_version=`$PKG_CONFIG --modversion openscenegraph-osg`
                    AC_MSG_NOTICE(OpenSceneGraph version: $osg_version)
                    $3
                ], [
                    AC_MSG_WARN(OpenSceneGraph required version: >= $2)
                    $4
                ])
        ],
        [$4])
])

# XCAM_CHECK_DNN([$1:value], [$2:inc-path], [$3:libs-path], [$4:if-found], [$5:if-not-found])
AC_DEFUN([XCAM_CHECK_DNN],
[
    AS_IF([test "x$1" = "xyes"],
        [
            AS_IF([test -z $2 || test -z $3],
                [
                    AC_MSG_WARN(Please export OPENVINO_IE_INC_PATH and OPENVINO_IE_LIBS_PATH environment variables)
                    AC_MSG_ERROR(OpenVino inc-path and libs-path have not been set!)
                ],
                [$4])
        ],
        [$5])
])

# XCAM_CHECK_AIQ([$1:value], [$2:if-found], [$3:if-not-found], [$4:if-found-local], [$5:if-not-found-local])
AC_DEFUN([XCAM_CHECK_AIQ],
[
    AS_IF([test "x$1" = "xyes"],
        [
            $2
            local_aiq=0
            PKG_CHECK_MODULES(IA_AIQ, ia_imaging,
                [
                    PKG_CHECK_EXISTS(ia_imaging >= 2.7,
                        AC_DEFINE(HAVE_AIQ_2_7, 1, [defined if module ia_imaging >= v2.0_007 is found]))
                ], [
                    local_aiq=1
                ])

            AS_IF([test "x$local_aiq" = "x1"],
                [
                    $4

                    #check HAVE_AIQ_2_7
                    AC_CACHE_CHECK([for ext/ia_imaging >=v2.0_007],
                        ac_cv_have_aiq_2_7, [
                        saved_CPPFLAGS="$CPPFLAGS"
                        saved_LIBS="$LIBS"

                        CPPFLAGS="$CPPFLAGS -I./ext/ia_imaging/include"
                        LIBS="$LIBS"
                        AC_COMPILE_IFELSE(
                            [AC_LANG_PROGRAM(
                                [[#include <stdint.h>
                                  #include <stdio.h>
                                  #include "ia_aiq_types.h"
                                ]],
                                [[ia_aiq_ae_results ae_result;
                                  ae_result.flashes = NULL;
                                ]]
                               )],
                            [ac_cv_have_aiq_2_7="yes"],
                            [ac_cv_have_aiq_2_7="no"]
                        )
                        CPPFLAGS="$saved_CPPFLAGS"
                        LIBS="$saved_LIBS"
                    ])

                    AS_IF([test "x$ac_cv_have_aiq_2_7" = "xyes"],
                        AC_DEFINE(HAVE_AIQ_2_7, 1, [defined if ia_imaging >= v2.0_007 is found]), [])
                ], [$5])

            AS_IF([test "x$local_aiq" = "x0"],
                [
                    IA_IMAGING_CFLAGS="$IA_AIQ_CFLAGS"
                    IA_IMAGING_LIBS="$IA_AIQ_LIBS"
                ], [
                    IA_IMAGING_CFLAGS="-I\$(top_srcdir)/ext/ia_imaging/include"
                    IA_IMAGING_LIBS="-L\$(top_srcdir)/ext/ia_imaging/lib -lia_aiq -lia_isp_2_2 -lia_cmc_parser -lia_mkn -lia_nvm -lia_exc -lia_log"
                ])
            AC_SUBST(IA_IMAGING_CFLAGS)
            AC_SUBST(IA_IMAGING_LIBS)
            LIBS="$LIBS $IA_IMAGING_LIBS"
        ], [
            $3
            $5
        ])
])

# XCAM_CHECK_LOCAL_ATOMISP([$1:value], [$2:if-found], [$3:if-not-found])
AC_DEFUN([XCAM_CHECK_LOCAL_ATOMISP],
[
    AS_IF([test "x$1" = "xyes"],
        [
            AC_MSG_NOTICE(update atomisp submodule)
            git submodule sync
            git submodule init
            git submodule update

            AC_CACHE_CHECK([for linux/atomisp.h],
                ac_cv_have_atomisp_headers, [
                saved_CPPFLAGS="$CPPFLAGS"
                saved_LIBS="$LIBS"

                CPPFLAGS="$CPPFLAGS"
                LIBS="$LIBS"
                AC_COMPILE_IFELSE(
                    [AC_LANG_PROGRAM(
                        [[#ifndef __user
                          #define __user
                          #endif
                          #include <stdint.h>
                          #include <stdio.h>
                          #include <linux/atomisp.h>]],
                        [[struct atomisp_parm param;]]
                       )],
                    [ac_cv_have_atomisp_headers="yes"],
                    [
                        ac_cv_have_atomisp_headers="no"
                        $2
                    ]
                )
                CPPFLAGS="$saved_CPPFLAGS"
                LIBS="$saved_LIBS"
            ])
        ],
        [$3])
])

# XCAM_CHECK_GST([$1:value], [$2:api-version], [$3:min-version], [$4:if-found], [$5:if-not-found])
AC_DEFUN([XCAM_CHECK_GST],
[
    AS_IF([test "x$1" = "xyes"],
        [
            $4
            PKG_CHECK_MODULES([GST], [gstreamer-$2 >= $3])
            PKG_CHECK_MODULES([GST_ALLOCATOR], [gstreamer-allocators-$2 >= $3])
            PKG_CHECK_MODULES([GST_VIDEO], [gstreamer-video-$2 >= $3])
        ],
        [$5])
])

# XCAM_MD5SUM([$1:file], [$2:md5sum], [$3:if-true], [$4:if-false])
AC_DEFUN([XCAM_MD5SUM],
[
    AS_IF([test -f $1],
        [
            md5=`md5sum $1 | cut --delimiter=' ' --fields=1`
            AS_IF([test "x$md5" = "x$2"], [$3], [$4])
        ],
        [$4])
])

# XCAM_WGET([$1:url], [$2:output-file], [$3:md5sum])
AC_DEFUN([XCAM_WGET],
[
    MD5_CORRECT=yes
    XCAM_MD5SUM([$2], [$3], [MD5_CORRECT=yes], [MD5_CORRECT=no])
    AS_IF([test "x$MD5_CORRECT" = "xyes"],
        [AC_MSG_NOTICE([checking $2 md5sum ... ok])],
        [
            AC_MSG_NOTICE([downloading $2...])
            dir=`dirname $2`
            AS_IF([test ! -d $dir], [mkdir -p $dir])

            wget --tries=2 --timeout=5 -q --no-use-server-timestamps $1 -O $2
            AS_IF([test "$?" != 0], [AC_MSG_ERROR([download/wget $2 failed])])

            XCAM_MD5SUM([$2], [$3], [AC_MSG_NOTICE([checking $2 md5sum ... ok])], [AC_MSG_ERROR([checking $2 md5sum ... failed])])
        ])
])

# XCAM_CHECK_PKG_CONFIG([$1:variable], [$2:value], [$3:if-true])
AC_DEFUN([XCAM_CHECK_PKG_CONFIG],
[
    AS_IF([test "x$1" = "x$2"], [$3], [])
])

