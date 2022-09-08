lib_LTLIBRARIES = libxcam_capi.la

XCAMCAPI_CXXFLAGS = \
    $(XCAM_CXXFLAGS)        \
    -I$(top_srcdir)/xcore   \
    -I$(top_srcdir)/modules \
    $(NULL)

XCAMCAPI_LIBS = \
    $(top_builddir)/xcore/libxcam_core.la        \
    $(top_builddir)/modules/soft/libxcam_soft.la \
    $(NULL)

if HAVE_LIBCL
XCAMCAPI_CXXFLAGS += $(LIBCL_CFLAGS)
XCAMCAPI_LIBS += \
    $(LIBCL_LIBS) -ldl                         \
    $(top_builddir)/modules/ocl/libxcam_ocl.la \
    $(NULL)
endif

if HAVE_GLES
XCAMCAPI_LIBS += $(top_builddir)/modules/gles/libxcam_gles.la
endif

if HAVE_VULKAN
XCAMCAPI_LIBS += $(top_builddir)/modules/vulkan/libxcam_vulkan.la
endif

if HAVE_LIBDRM
XCAMCAPI_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMCAPI_LIBS += \
    -ldrm_intel    \
    $(LIBDRM_LIBS) \
    $(NULL)
endif

xcam_ocl_sources = \
    xcam_handle.cpp         \
    context_priv.cpp        \
    ctxs/context_stitch.cpp \
    $(NULL)

if HAVE_LIBCL
xcam_ocl_sources += ctxs/context_cl.cpp
endif

libxcam_capi_la_SOURCES = \
    $(xcam_ocl_sources) \
    $(NULL)

libxcam_capi_la_CXXFLAGS = \
    $(XCAMCAPI_CXXFLAGS) \
    $(NULL)

libxcam_capi_la_LIBADD = \
    $(XCAMCAPI_LIBS) \
    $(NULL)

libxcam_capi_la_LDFLAGS = \
    $(XCAM_LT_LDFLAGS) \
    $(NULL)

libxcam_capiincludedir = $(includedir)/xcam/capi

nobase_libxcam_capiinclude_HEADERS = \
    xcam_handle.h \
    $(NULL)

libxcam_capi_la_LIBTOOLFLAGS = --tag=disable-static
