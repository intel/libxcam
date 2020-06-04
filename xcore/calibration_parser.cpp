/*
 * calibration_parser.cpp - parse fisheye calibration file
 *
 *  Copyright (c) 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Junkai Wu <junkai.wu@intel.com>
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "calibration_parser.h"
#include "file_handle.h"
#include <unistd.h>

#if HAVE_JSON
#include <fstream>
#include <json.hpp>
using json = nlohmann::json;
#endif

namespace XCam {

CalibrationParser::CalibrationParser()
{
}

#define CHECK_NULL(ptr) \
    if (ptr == NULL) { \
        XCAM_LOG_ERROR("Parse file failed"); \
        return XCAM_RETURN_ERROR_FILE; \
    }

XCamReturn
CalibrationParser::parse_intrinsic_param(char *file_body, IntrinsicParameter &intrinsic_param)
{
    char *line_str = NULL;
    char *line_endptr = NULL;
    char *tok_str = NULL;
    char *tok_endptr = NULL;
    static const char *line_tokens = "\r\n";
    static const char *str_tokens = " \t";

    do {
        line_str = strtok_r(file_body, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.poly_length = strtol(tok_str, NULL, 10);

        XCAM_FAIL_RETURN (
            ERROR, intrinsic_param.poly_length <= XCAM_INTRINSIC_MAX_POLY_SIZE,
            XCAM_RETURN_ERROR_PARAM,
            "intrinsic poly length:%d is larger than max_size:%d.",
            intrinsic_param.poly_length, XCAM_INTRINSIC_MAX_POLY_SIZE);

        for (uint32_t i = 0; i < intrinsic_param.poly_length; i++) {
             tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
             CHECK_NULL(tok_str);
             intrinsic_param.poly_coeff[i] = (strtof(tok_str, NULL));
        }

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.cy = strtof(tok_str, NULL);

        tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
        CHECK_NULL(tok_str);
        intrinsic_param.cx = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.c = strtof(tok_str, NULL);

        tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
        CHECK_NULL(tok_str);
        intrinsic_param.d = strtof(tok_str, NULL);

        tok_str = strtok_r(NULL, str_tokens, &tok_endptr);
        CHECK_NULL(tok_str);
        intrinsic_param.e = strtof(tok_str, NULL);
    } while(0);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CalibrationParser::parse_extrinsic_param(char *file_body, ExtrinsicParameter &extrinsic_param)
{
    char *line_str = NULL;
    char *line_endptr = NULL;
    char *tok_str = NULL;
    char *tok_endptr = NULL;
    static const char *line_tokens = "\r\n";
    static const char *str_tokens = " \t";

    do {
        line_str = strtok_r(file_body, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_x = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_y = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_z = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.roll = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.pitch = strtof(tok_str, NULL);

        line_str = strtok_r(NULL, line_tokens, &line_endptr);
        CHECK_NULL(line_str);
        tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
               line_str = strtok_r(NULL, line_tokens, &line_endptr);
               CHECK_NULL(line_str);
               tok_str = strtok_r(line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.yaw = strtof(tok_str, NULL);
    } while(0);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CalibrationParser::parse_intrinsic_file(const char *file_path, IntrinsicParameter &intrinsic_param)
{
    XCAM_FAIL_RETURN (
        ERROR, !access (file_path, R_OK), XCAM_RETURN_ERROR_PARAM,
        "cannot access intrinsic file %s", file_path);

    FileHandle file_reader;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    std::vector<char> context;
    size_t file_size = 0;

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret = file_reader.open (file_path, "r")), ret,
        "open intrinsic file(%s) failed.", file_path);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret = file_reader.get_file_size (file_size)), ret,
        "read intrinsic file(%s) failed to get file size.", file_path);
    context.resize (file_size + 1);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret = file_reader.read_file (&context[0], file_size)), ret,
        "read intrinsic file(%s) failed, file size:%d.", file_path, (int)file_size);
    file_reader.close ();
    context[file_size] = '\0';

    return parse_intrinsic_param (&context[0], intrinsic_param);
}

XCamReturn
CalibrationParser::parse_extrinsic_file(const char *file_path, ExtrinsicParameter &extrinsic_param)
{
    XCAM_FAIL_RETURN (
        ERROR, !access (file_path, R_OK), XCAM_RETURN_ERROR_PARAM,
        "cannot access extrinsic file %s", file_path);

    FileHandle file_reader;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    std::vector<char> context;
    size_t file_size = 0;

    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.open (file_path, "r")), ret,
        "open extrinsic file(%s) failed.", file_path);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.get_file_size (file_size)), ret,
        "read extrinsic file(%s) failed to get file size.", file_path);
    context.resize (file_size + 1);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.read_file (&context[0], file_size)), ret,
        "read extrinsic file(%s) failed, file size:%d.", file_path, (int)file_size);
    file_reader.close ();
    context[file_size] = '\0';

    return parse_extrinsic_param (&context[0], extrinsic_param);
}

#if HAVE_JSON
XCamReturn
CalibrationParser::parse_fisheye_camera_param (const char *file_path, FisheyeInfo *fisheye_info, uint32_t camera_count)
{
    XCAM_LOG_DEBUG ("Parse camera calibration file: %s", file_path);
    if (NULL == file_path) {
        XCAM_LOG_ERROR ("invalide input file path !");
        return XCAM_RETURN_ERROR_PARAM;
    }
    std::ifstream calibFile(file_path);
    if (!calibFile.is_open()) {
        XCAM_LOG_ERROR ("calibration file Not Found!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    try {
        json calib_params = json::parse(calibFile);

        auto const rig = calib_params.find("rig");
        if (rig == calib_params.end()) {
            XCAM_LOG_ERROR ("rig Not Found");
        }

        auto const model = rig->find("model");
        if (model != rig->end()) {
            XCAM_LOG_DEBUG ("camera model=%d ", rig->find("model")->get<int>());
        } else {
            XCAM_LOG_WARNING ("model Not Found");
        }

        auto const cameras = rig->find("cameras");
        if (cameras == rig->end()) {
            XCAM_LOG_ERROR ("cameras Not Found");
            return XCAM_RETURN_ERROR_PARAM;
        }

        auto const camera = cameras->find("camera");
        if (camera == cameras->end()) {
            XCAM_LOG_ERROR ("camera Not Found");
            return XCAM_RETURN_ERROR_PARAM;
        }

        uint32_t cam_id = 0;
        for (json::iterator cam = camera->begin(); cam != camera->end(), cam_id < camera_count; cam++) {
            fisheye_info[cam_id].intrinsic.fx = cam->find("fx")->get<float>();
            fisheye_info[cam_id].intrinsic.fy = cam->find("fy")->get<float>();
            fisheye_info[cam_id].intrinsic.cx = cam->find("cx")->get<float>();
            fisheye_info[cam_id].intrinsic.cy = cam->find("cy")->get<float>();
            fisheye_info[cam_id].intrinsic.width = cam->find("w")->get<int>();
            fisheye_info[cam_id].intrinsic.height = cam->find("h")->get<int>();
            fisheye_info[cam_id].intrinsic.skew = cam->find("skew")->get<float>();
            fisheye_info[cam_id].intrinsic.fov = cam->find("fov")->get<float>();
            fisheye_info[cam_id].intrinsic.flip = (strcasecmp(cam->find("flip")->get<std::string>().c_str(), "true") == 0 ? true : false);
            XCAM_LOG_DEBUG ("cam[%d]: flip=%d ", cam_id, fisheye_info[cam_id].intrinsic.flip);
            XCAM_LOG_DEBUG ("fx=%f ", fisheye_info[cam_id].intrinsic.fx);
            XCAM_LOG_DEBUG ("fy=%f ", fisheye_info[cam_id].intrinsic.fy);
            XCAM_LOG_DEBUG ("cx=%f ", fisheye_info[cam_id].intrinsic.cx);
            XCAM_LOG_DEBUG ("cy=%f ", fisheye_info[cam_id].intrinsic.cy);
            XCAM_LOG_DEBUG ("w=%d ", fisheye_info[cam_id].intrinsic.width);
            XCAM_LOG_DEBUG ("h=%d ", fisheye_info[cam_id].intrinsic.height);
            XCAM_LOG_DEBUG ("fov=%f ", fisheye_info[cam_id].intrinsic.fov);
            XCAM_LOG_DEBUG ("skew=%f ", fisheye_info[cam_id].intrinsic.skew);

            auto const k = cam->find("K");
            uint32_t i = 0;
            for (json::iterator k_mat = k->begin(); k_mat != k->end(); k_mat++, i++) {
                XCAM_LOG_DEBUG ("k[%d]: %f ", i, k_mat->get<float>());
            }

            auto const d = cam->find("D");
            i = 0;
            for (json::iterator d_vec = d->begin(); d_vec != d->end(), i < 4; d_vec++, i++) {
                fisheye_info[cam_id].distort_coeff[i] = d_vec->get<float>();
                XCAM_LOG_DEBUG ("d[%d]: %f ", i, d_vec->get<float>());
            }

            auto const r = cam->find("R");
            Mat3f rotation;
            i = 0;
            for (json::iterator r_mat = r->begin(); r_mat != r->end(), i < 9; r_mat++, i++) {
                rotation(i / 3, i % 3) = r_mat->get<float>();
            }
            Quaternion<float> quat = create_quaternion (rotation);
            Vec3f euler_angles = quat.euler_angles ();
            fisheye_info[cam_id].extrinsic.yaw = RADIANS_2_DEGREE (euler_angles[0]);
            fisheye_info[cam_id].extrinsic.pitch = RADIANS_2_DEGREE (euler_angles[1]);
            fisheye_info[cam_id].extrinsic.roll = RADIANS_2_DEGREE (euler_angles[2]);

            auto const t = cam->find("t");
            i = 0;
            Vec3f translation;
            for (json::iterator t_vec = t->begin(); t_vec != t->end(), i < 3; t_vec++, i++) {
                translation[i] = t_vec->get<float>();
                XCAM_LOG_DEBUG ("t[%d]: %f ", i, t_vec->get<float>());
            }
            fisheye_info[cam_id].extrinsic.trans_x = translation[0];
            fisheye_info[cam_id].extrinsic.trans_y = translation[1];
            fisheye_info[cam_id].extrinsic.trans_z = translation[2];

            auto const c = cam->find("c");
            i = 0;
            for (json::iterator c_vec = c->begin(); c_vec != c->end(), i < 3; c_vec++, i++) {
                fisheye_info[cam_id].c_coeff[i] = c_vec->get<float>();
                XCAM_LOG_DEBUG ("c[%d]: %f ", i, c_vec->get<float>());
            }

            cam_id++;
        }
    } catch (std::exception&) {
        XCAM_LOG_ERROR ("parse camera calibration JSON file failed!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    return XCAM_RETURN_NO_ERROR;
}
#endif

}
