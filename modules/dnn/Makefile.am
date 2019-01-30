lib_LTLIBRARIES = libxcam_dnn.la

XCAM_DNN_CXXFLAGS = \
    $(XCAM_CXXFLAGS)        \
    -I$(top_srcdir)/xcore   \
    -I$(top_srcdir)/modules \
    $(OPENVINO_IE_CXXFLAGS) \
    $(NULL)

XCAM_DNN_LIBS = \
    $(top_builddir)/xcore/libxcam_core.la \
    -L$(OPENVINO_IE_LIBS) \
    -linference_engine \
    -lclDNN64 \
    -lclDNNPlugin \
    -lcpu_extension \
    -lMKLDNNPlugin \
    -lformat_reader \
    $(NULL)

xcam_dnn_sources =       \
    dnn_inference_engine.cpp   \
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
    -Wl,-rpath,$(OPENVINO_IE_LIBS) \
    $(NULL)

libxcam_dnnincludedir = $(includedir)/xcam/dnn

nobase_libxcam_dnninclude_HEADERS = \
    dnn_inference_engine.h          \
    $(NULL)

libxcam_dnn_la_LIBTOOLFLAGS = --tag=disable-static
