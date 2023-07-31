lib_LTLIBRARIES = libxcam_dnn.la

XCAM_DNN_CXXFLAGS = \
    $(XCAM_CXXFLAGS)        \
    $(LIBOPENVINO_CFLAGS)   \
    -I$(top_srcdir)/xcore   \
    -I$(top_srcdir)/modules \
    -I$(OPENVINO_EXTRAS_PATH)/samples/cpp/common/format_reader/include \
    $(NULL)

XCAM_DNN_LIBS = \
    $(top_builddir)/xcore/libxcam_core.la \
    -L$(OPENVINO_EXTRAS_PATH)/bin/intel64/Release \
    -lopenvino \
    -lopenvino_intel_gpu_plugin \
    -lopenvino_intel_cpu_plugin \
    -lopenvino_intel_gna_plugin \
    -lopenvino_hetero_plugin \
    -lformat_reader \
    -lie_samples_utils \
    $(NULL)

if HAVE_OPENCV
XCAM_DNN_CXXFLAGS += $(OPENCV_CFLAGS)
XCAM_DNN_LIBS += $(top_builddir)/modules/ocv/libxcam_ocv.la
endif

xcam_dnn_sources =       \
    dnn_inference_engine.cpp       \
    dnn_object_detection.cpp       \
    dnn_super_resolution.cpp       \
    dnn_semantic_segmentation.cpp  \
    dnn_inference_utils.cpp        \
    $(NULL)

libxcam_dnn_la_SOURCES = \
    $(xcam_dnn_sources)  \
    $(NULL)

libxcam_dnn_la_CXXFLAGS = \
    $(XCAM_DNN_CXXFLAGS)  \
    $(NULL)

libxcam_dnn_la_LIBADD = \
    $(XCAM_DNN_LIBS)    \
    $(NULL)

libxcam_dnn_la_LDFLAGS = \
    $(XCAM_LT_LDFLAGS)   \
    -Wl,-rpath,$(OPENVINO_IE_LIBS_PATH) \
    $(NULL)

libxcam_dnnincludedir = $(includedir)/xcam/dnn

nobase_libxcam_dnninclude_HEADERS = \
    dnn_inference_engine.h          \
    dnn_object_detection.h          \
    dnn_super_resolution.h          \
    dnn_semantic_segmentation.h     \
    dnn_inference_utils.h           \
    $(NULL)

libxcam_dnn_la_LIBTOOLFLAGS = --tag=disable-static
