SUBDIRS = interface

lib_LTLIBRARIES = libstub.la

PTHREAD_LDFLAGS = -pthread

STUB_CXXFLAGS = -fPIC -std=c++11
STUB_LIBS = \
		   $(NULL)

XCAMSRC_CXXFLAGS = -fPIC -std=c++11
XCAMSRC_LIBS = \
		   $(NULL)

if HAVE_LIBDRM
STUB_CXXFLAGS += $(LIBDRM_CFLAGS)
STUB_LIBS += $(LIBDRM_LIBS)
endif

if HAVE_LIBDRM
XCAMSRC_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAMSRC_LIBS += $(LIBDRM_LIBS)
endif


libstub_la_SOURCES = stub.cpp bufmap.cpp v4l2dev.cpp fmt.cpp

libstub_la_CXXFLAGS = -I$(top_builddir)/xcore -I$(top_builddir)/xcore/base \
			$(GST_CFLAGS) \
			-I$(top_builddir)/wrapper/gstreamer/interface \
			 $(STUB_CXXFLAGS)	\
			 $(NULL)    
libstub_la_LDFLAGS = \
			-static	\
			$(PTHREAD_LDFLAGS)  \
			$(NULL)

libstub_la_LIBADD = $(top_builddir)/xcore/libxcam_core.la \
		       $(STUB_LIBS)  \
		       -lpthread \
		       $(NULL)

noinst_HEADERS = stub.h bufmap.h v4l2dev.h fmt.h

# Note: plugindir is set in configure
plugindir="\$(libdir)/gstreamer-1.0"

plugin_LTLIBRARIES = libgstxcamsrc.la

# sources used to compile this plug-in
libgstxcamsrc_la_SOURCES = gstxcambufferpool.cpp	\
			   gstxcamsrc.cpp

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstxcamsrc_la_CXXFLAGS = -I$(top_builddir)/xcore -I$(top_builddir)/xcore/base   \
			   $(GST_CFLAGS) $(XCAMSRC_CXXFLAGS)	\
			  -I$(top_builddir)/wrapper/gstreamer \
	                  -I$(top_builddir)/wrapper/gstreamer/interface


libgstxcamsrc_la_LIBADD = $(XCAMSRC_LIBS) \
			  $(top_builddir)/wrapper/gstreamer/libstub.la	\
			  $(top_builddir)/wrapper/gstreamer/interface/libgstxcaminterface.la	    \
			  $(GST_ALLOCATOR_LIBS)    \
			  $(GST_VIDEO_LIBS)        \
			  $(GST_LIBS)

libgstxcamsrc_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstxcamsrc_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS += gstxcambufferpool.h	\
		 gstxcamsrc.h
