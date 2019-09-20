noinst_LTLIBRARIES = libxcam_dvs.la

libxcam_dvs_la_SOURCES = \
    libdvs.cpp     \
    stabilizer.cpp \
    $(NULL)

libxcam_dvs_la_CXXFLAGS = \
    $(XCAM_CXXFLAGS)      \
    $(OPENCV_CFLAGS)      \
    -I$(top_srcdir)/xcore \
    $(NULL)

libxcam_dvs_la_LIBADD = \
    $(top_builddir)/xcore/libxcam_core.la \
    $(NULL)

noinst_PROGRAMS = test_image_stabilization

test_image_stabilization_SOURCES = \
    test-image-stabilization.cpp \
    $(NULL)

test_image_stabilization_CXXFLAGS = \
    $(XCAM_CXXFLAGS)      \
    -I$(top_srcdir)/xcore \
    $(NULL)

test_image_stabilization_LDADD = \
    libxcam_dvs.la \
    $(OPENCV_LIBS) \
    $(NULL)
