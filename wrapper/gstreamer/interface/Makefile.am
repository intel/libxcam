lib_LTLIBRARIES = libgstxcaminterface.la

libgstxcaminterface_la_SOURCES = gstxcaminterface.c

GST_CFLAGS = -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/usr/lib/glib-2.0/include
LIBXCAM_CFLAGS = -I/usr/include/xcam -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include 

libgstxcaminterface_la_CFLAGS = \
				 $(GST_CFLAGS) \
				 $(LIBXCAM_CFLAGS)

libgstxcaminterface_la_LIBADD = \
				 $(GST_LIBS) \
				 $(NULL)

libgstxcaminterfaceincludedir = \
	$(includedir)/gstreamer-1.0/gst

libgstxcaminterfaceinclude_HEADERS = \
	gstxcaminterface.h

