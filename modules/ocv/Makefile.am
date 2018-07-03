noinst_LIBRARIES = libxcam_ocv.a

XCAMOCV_CXXFLAGS = \
    $(XCAM_CXXFLAGS)        \
    -I$(top_srcdir)/xcore   \
    -I$(top_srcdir)/modules \
    $(NULL)

XCAMOCV_LIBS = \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

if HAVE_OPENCV
XCAMOCV_CXXFLAGS += $(OPENCV_CFLAGS)
endif

xcam_ocv_sources = \
    cv_context.cpp               \
    cv_base_class.cpp            \
    cv_image_process_helper.cpp  \
    cv_image_sharp.cpp           \
    cv_edgetaper.cpp             \
    cv_wiener_filter.cpp         \
    cv_image_deblurring.cpp      \
    cv_feature_match.cpp         \
    cv_feature_match_cluster.cpp \
    cv_capi_feature_match.cpp    \
    $(NULL)

libxcam_ocv_a_SOURCES = \
    $(xcam_ocv_sources) \
    $(NULL)

libxcam_ocv_a_CXXFLAGS = \
    $(XCAMOCV_CXXFLAGS) \
    $(NULL)

libxcam_ocv_a_LIBADD = \
    $(XCAMOCV_LIBS) \
    $(NULL)
