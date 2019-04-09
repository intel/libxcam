noinst_PROGRAMS = \
    test-soft-image     \
    test-surround-view  \
    test-device-manager \
    $(NULL)

if HAVE_LIBCL
noinst_PROGRAMS += \
    test-cl-image            \
    test-image-blend         \
    test-pipe-manager        \
    test-binary-kernel       \
    test-image-stitching     \
    test-video-stabilization \
    $(NULL)
endif

if HAVE_GLES
noinst_PROGRAMS += \
    test-gles-handler \
    $(NULL)
endif

if HAVE_VULKAN
noinst_PROGRAMS += \
    test-vk-handler \
    $(NULL)
endif

if ENABLE_RENDER
noinst_PROGRAMS += \
    test-render-surround-view \
    $(NULL)
endif

if ENABLE_DNN
noinst_PROGRAMS += \
    test-dnn-inference \
    $(NULL)
endif

if HAVE_OPENCV
noinst_PROGRAMS += \
    test-image-deblurring \
    $(NULL)
endif

TEST_BASE_CXXFLAGS = \
    $(XCAM_CXXFLAGS)        \
    -I$(top_srcdir)/xcore   \
    -I$(top_srcdir)/modules \
    $(NULL)

TEST_CORE_LA = $(top_builddir)/xcore/libxcam_core.la

if HAVE_OPENCV
TEST_OCV_LA = $(top_builddir)/modules/ocv/libxcam_ocv.la
endif

test_device_manager_SOURCES = test-device-manager.cpp
test_device_manager_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_device_manager_LDADD = $(TEST_CORE_LA)
if USE_LOCAL_ATOMISP
test_device_manager_CXXFLAGS += -I$(top_srcdir)/ext/atomisp
endif
if ENABLE_IA_AIQ
test_device_manager_LDADD += $(top_builddir)/modules/isp/libxcam_isp.la
endif

if HAVE_LIBCL
test_device_manager_LDADD += $(top_builddir)/modules/ocl/libxcam_ocl.la

TEST_OCL_LA = $(top_builddir)/modules/ocl/libxcam_ocl.la

test_cl_image_SOURCES = test-cl-image.cpp
test_cl_image_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_cl_image_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCL_LA)  \
    $(NULL)

test_binary_kernel_SOURCES = test-binary-kernel.cpp
test_binary_kernel_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_binary_kernel_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCL_LA)  \
    $(NULL)

test_pipe_manager_SOURCES = test-pipe-manager.cpp
test_pipe_manager_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_pipe_manager_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCL_LA)  \
    $(NULL)

test_image_blend_SOURCES = test-image-blend.cpp
test_image_blend_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_image_blend_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCL_LA)  \
    $(NULL)

test_image_stitching_SOURCES = test-image-stitching.cpp
test_image_stitching_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_image_stitching_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCL_LA)  \
    $(TEST_OCV_LA)  \
    $(NULL)

test_video_stabilization_SOURCES = test-video-stabilization.cpp
test_video_stabilization_CXXFLAGS = \
    $(TEST_BASE_CXXFLAGS) \
    $(OPENCV_CFLAGS)      \
    $(NULL)
test_video_stabilization_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCL_LA)  \
    $(TEST_OCV_LA)  \
    $(OPENCV_LIBS)  \
    $(NULL)
endif

if HAVE_OPENCV
test_image_deblurring_SOURCES = test-image-deblurring.cpp
test_image_deblurring_CXXFLAGS = \
    $(TEST_BASE_CXXFLAGS) \
    $(OPENCV_CFLAGS)      \
    $(NULL)
test_image_deblurring_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCV_LA)  \
    $(OPENCV_LIBS)  \
    $(NULL)
endif

TEST_SOFT_LA = $(top_builddir)/modules/soft/libxcam_soft.la

test_soft_image_SOURCES = test-soft-image.cpp
test_soft_image_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_soft_image_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCV_LA)  \
    $(TEST_SOFT_LA) \
    $(NULL)

if HAVE_GLES
TEST_GLES_LA = $(top_builddir)/modules/gles/libxcam_gles.la
endif

if HAVE_VULKAN
TEST_VK_LA = $(top_builddir)/modules/vulkan/libxcam_vulkan.la
endif

test_surround_view_SOURCES = test-surround-view.cpp
test_surround_view_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_surround_view_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCV_LA)  \
    $(TEST_SOFT_LA) \
    $(TEST_GLES_LA) \
    $(TEST_VK_LA)   \
    $(NULL)

if HAVE_VULKAN
test_vk_handler_SOURCES = test-vk-handler.cpp
test_vk_handler_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_vk_handler_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCV_LA)  \
    $(TEST_VK_LA)   \
    $(NULL)
endif

if HAVE_GLES
test_gles_handler_SOURCES = test-gles-handler.cpp
test_gles_handler_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_gles_handler_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_OCV_LA)  \
    $(TEST_GLES_LA) \
    $(NULL)
endif

if ENABLE_RENDER
TEST_RENDER_LA = $(top_builddir)/modules/render/libxcam_render.la

test_render_surround_view_SOURCES = test-render-surround-view.cpp
test_render_surround_view_CXXFLAGS = $(TEST_BASE_CXXFLAGS)
test_render_surround_view_LDADD = \
    $(TEST_CORE_LA)   \
    $(TEST_OCV_LA)    \
    $(TEST_SOFT_LA)   \
    $(TEST_GLES_LA)   \
    $(TEST_VK_LA)     \
    $(TEST_RENDER_LA) \
    $(NULL)
endif

if ENABLE_DNN
TEST_DNN_LA = $(top_builddir)/modules/dnn/libxcam_dnn.la

test_dnn_inference_SOURCES = test-dnn-inference.cpp
test_dnn_inference_CXXFLAGS = \
    $(TEST_BASE_CXXFLAGS)                   \
    -I$(OPENVINO_IE_INC_PATH)/include       \
    -I$(OPENVINO_IE_INC_PATH)/src/extension \
    $(NULL)

test_dnn_inference_LDADD = \
    $(TEST_CORE_LA) \
    $(TEST_DNN_LA)  \
    $(NULL)
endif

