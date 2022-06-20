glslx_sources = \
    shader_copy.comp.slx                     \
    shader_geomap.comp.slx                   \
    shader_gauss_scale_pyr.comp.slx          \
    shader_lap_trans_pyr.comp.slx            \
    shader_blend_pyr.comp.slx                \
    shader_reconstruct_pyr.comp.slx          \
    shader_geomap_yuv420.comp.slx            \
    shader_gauss_scale_pyr_yuv420.comp.slx   \
    shader_lap_trans_pyr_yuv420.comp.slx     \
    shader_blend_pyr_yuv420.comp.slx         \
    shader_reconstruct_pyr_yuv420.comp.slx   \
    shader_geomap_fastmap_y.comp.slx         \
    shader_geomap_fastmap_uv_nv12.comp.slx   \
    shader_geomap_fastmap_uv_yuv420.comp.slx \
    shader_fastmap_blend_y.comp.slx          \
    shader_fastmap_blend_uv_nv12.comp.slx    \
    shader_fastmap_blend_uv_yuv420.comp.slx  \
    shader_copy_tex_to_ssbo.comp.slx         \
    shader_copy_ssbo_to_tex.comp.slx         \
    $(NULL)

add_quotation_marks_sh = \
    $(top_srcdir)/tools/add-quotation-marks.sh

glsl_dir = $(top_srcdir)/shaders/glsl

all-local: $(glslx_sources)

$(glslx_sources): %.slx: $(glsl_dir)/%.sl
	@$(add_quotation_marks_sh) $< $@

CLEANFILES = $(glslx_sources)
