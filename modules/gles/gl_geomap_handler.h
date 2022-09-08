/*
 * gl_geomap_handler.h - gl geometry map handler class
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_GL_GEOMAP_HANDER_H
#define XCAM_GL_GEOMAP_HANDER_H

#include <interface/geo_mapper.h>
#include <gles/gl_image_shader.h>
#include <gles/gl_image_handler.h>

namespace XCam {

namespace GLGeoMapPriv {
class ComMap;
class FastMap;
}

class GLGeoMapHandler
    : public GLImageHandler, public GeoMapper
{
public:
    GLGeoMapHandler (const char *name = "GLGeoMapHandler");
    ~GLGeoMapHandler ();

    bool set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height);
    void get_left_factors (float &x, float &y);
    void get_right_factors (float &x, float &y);

    bool set_std_area (const Rect &area);
    const Rect &get_std_area () const;

    bool set_extended_offset (uint32_t offset);
    uint32_t get_extended_offset () const;

    bool set_lut_buf (const SmartPtr<GLBuffer> &buf);
    const SmartPtr<GLBuffer> &get_lut_buf () const;

    bool set_coordx_y (const SmartPtr<GLBuffer> &coordx_y);
    const SmartPtr<GLBuffer> &get_coordx_y () const;

    bool set_coordy_y (const SmartPtr<GLBuffer> &coordy_y);
    const SmartPtr<GLBuffer> &get_coordy_y () const;

    bool set_coordx_uv (const SmartPtr<GLBuffer> &coordx_uv);
    const SmartPtr<GLBuffer> &get_coordx_uv () const;

    bool set_coordy_uv (const SmartPtr<GLBuffer> &coordy_uv);
    const SmartPtr<GLBuffer> &get_coordy_uv () const;

    const float *get_lut_step () const;

    bool init_factors ();
    virtual bool update_factors (
        float left_factor_x, float left_factor_y, float right_factor_x, float right_factor_y);

    void activate_fastmap ();

    XCamReturn remap (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf);
    virtual XCamReturn terminate ();

protected:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    XCAM_DEAD_COPY (GLGeoMapHandler);
    XCamReturn ensure_default_params ();

protected:
    SmartPtr<GLBuffer>                 _lut_buf;
    SmartPtr<GLBuffer>                 _coordx_y;
    SmartPtr<GLBuffer>                 _coordy_y;
    SmartPtr<GLBuffer>                 _coordx_uv;
    SmartPtr<GLBuffer>                 _coordy_uv;

    float                              _lut_step[4];
    float                              _left_factor_x;
    float                              _left_factor_y;
    float                              _right_factor_x;
    float                              _right_factor_y;
    Rect                               _std_area;
    uint32_t                           _extended_offset;

    bool                               _activate_fastmap;
    bool                               _fastmap_activated;

    SmartPtr<GLGeoMapPriv::ComMap>     _commapper;
    SmartPtr<GLGeoMapPriv::FastMap>    _fastmapper;
};

class GLDualConstGeoMapHandler
    : public GLGeoMapHandler
{
public:
    explicit GLDualConstGeoMapHandler (const char *name = "GLDualConstGeoMapHandler")
        : GLGeoMapHandler (name)
    {}
    ~GLDualConstGeoMapHandler () {}

    virtual bool update_factors (
        float left_factor_x, float left_factor_y, float right_factor_x, float right_factor_y);
};

extern SmartPtr<GLImageHandler> create_gl_geo_mapper ();

}
#endif // XCAM_GL_GEOMAP_HANDER_H
