noinst_DATA = $(clx_kernel_SOURCES)

cl_quote_sh = $(top_srcdir)/tools/cl_double_quotation.sh
cl_kernel_dir = $(top_srcdir)/cl_kernel

clx_kernel_SOURCES =   \
	kernel_3a_stats.clx \
	kernel_blc.clx \
	kernel_csc_rgba64torgba.clx \
	kernel_csc_rgbatolab.clx \
	kernel_csc_rgbatonv12.clx \
	kernel_demo.clx \
	kernel_demosaic.clx \
	kernel_denoise.clx \
	kernel_gamma.clx \
	kernel_hdr_lab.clx \
	kernel_hdr_rgb.clx \
	kernel_macc.clx \
	kernel_snr.clx \
	kernel_tnr_rgb.clx \
	kernel_tnr_yuv.clx \
	kernel_wb.clx \
	$(NULL)

$(clx_kernel_SOURCES):
	@$(cl_quote_sh) $< $@

kernel_3a_stats.clx: $(cl_kernel_dir)/kernel_3a_stats.cl

kernel_blc.clx: $(cl_kernel_dir)/kernel_blc.cl

kernel_csc_rgba64torgba.clx: $(cl_kernel_dir)/kernel_csc_rgba64torgba.cl

kernel_csc_rgbatolab.clx: $(cl_kernel_dir)/kernel_csc_rgbatolab.cl

kernel_csc_rgbatonv12.clx: $(cl_kernel_dir)/kernel_csc_rgbatonv12.cl

kernel_demo.clx: $(cl_kernel_dir)/kernel_demo.cl

kernel_demosaic.clx: $(cl_kernel_dir)/kernel_demosaic.cl

kernel_denoise.clx: $(cl_kernel_dir)/kernel_denoise.cl

kernel_gamma.clx: $(cl_kernel_dir)/kernel_gamma.cl

kernel_hdr_lab.clx: $(cl_kernel_dir)/kernel_hdr_lab.cl

kernel_hdr_rgb.clx: $(cl_kernel_dir)/kernel_hdr_rgb.cl

kernel_macc.clx: $(cl_kernel_dir)/kernel_macc.cl

kernel_snr.clx: $(cl_kernel_dir)/kernel_snr.cl

kernel_tnr_rgb.clx: $(cl_kernel_dir)/kernel_tnr_rgb.cl

kernel_tnr_yuv.clx: $(cl_kernel_dir)/kernel_tnr_yuv.cl

kernel_wb.clx: $(cl_kernel_dir)/kernel_wb.cl

CLEANFILES = $(clx_kernel_SOURCES)
