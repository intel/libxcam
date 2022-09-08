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
#include "file.h"
#include <unistd.h>

#if HAVE_JSON
#include <fstream>
#include <json.hpp>
using json = nlohmann::json;
#endif

namespace XCam {

static const char *calib_attribute[] = {
    "camera_id",
    "K_matrix",
    "R_matrix",
    "T_matrix"
};

enum CalibAttribute {
    CalibCameraId = 0,
    CalibCameraMatrix,
    CalibRotationMatrix,
    CalibTranslationMatrix
};

CalibrationParser::CalibrationParser ()
{
}

#define CHECK_NULL(ptr) \
    if(ptr == NULL) { \
        XCAM_LOG_ERROR("Parse file failed"); \
        return XCAM_RETURN_ERROR_FILE; \
    }

#define CHECK_PARAM(ptr) \
    if(ptr == NULL) { \
        XCAM_LOG_DEBUG("Parse NULL param"); \
        continue; \
    }

XCamReturn
CalibrationParser::parse_intrinsic_param (char *file_body, IntrinsicParameter &intrinsic_param)
{
    char *line_str = NULL;
    char *line_endptr = NULL;
    char *tok_str = NULL;
    char *tok_endptr = NULL;
    static const char *line_tokens = "\r\n";
    static const char *str_tokens = " \t";

    do {
        line_str = strtok_r (file_body, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.poly_length = strtol (tok_str, NULL, 10);

        XCAM_FAIL_RETURN (
            ERROR, intrinsic_param.poly_length <= XCAM_INTRINSIC_MAX_POLY_SIZE,
            XCAM_RETURN_ERROR_PARAM,
            "intrinsic poly length:%d is larger than max_size:%d.",
            intrinsic_param.poly_length, XCAM_INTRINSIC_MAX_POLY_SIZE);

        for (uint32_t i = 0; i < intrinsic_param.poly_length; i++) {
            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            intrinsic_param.poly_coeff[i] = (strtof (tok_str, NULL));
        }

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.cy = strtof (tok_str, NULL);

        tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
        CHECK_NULL (tok_str);
        intrinsic_param.cx = strtof(tok_str, NULL);

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        intrinsic_param.c = strtof (tok_str, NULL);

        tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
        CHECK_NULL (tok_str);
        intrinsic_param.d = strtof (tok_str, NULL);

        tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
        CHECK_NULL (tok_str);
        intrinsic_param.e = strtof (tok_str, NULL);
    } while (0);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CalibrationParser::parse_extrinsic_param (char *file_body, ExtrinsicParameter &extrinsic_param)
{
    char *line_str = NULL;
    char *line_endptr = NULL;
    char *tok_str = NULL;
    char *tok_endptr = NULL;
    static const char *line_tokens = "\r\n";
    static const char *str_tokens = " \t";

    do {
        line_str = strtok_r (file_body, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_x = strtof (tok_str, NULL);

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL(line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_y = strtof (tok_str, NULL);

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.trans_z = strtof (tok_str, NULL);

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.roll = strtof (tok_str, NULL);

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.pitch = strtof (tok_str, NULL);

        line_str = strtok_r (NULL, line_tokens, &line_endptr);
        CHECK_NULL (line_str);
        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        while (tok_str == NULL || tok_str[0] == '#') {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            CHECK_NULL (line_str);
            tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        }
        extrinsic_param.yaw = strtof (tok_str, NULL);
    } while (0);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CalibrationParser::parse_intrinsic_file (const char *file_path, IntrinsicParameter &intrinsic_param)
{
    XCAM_FAIL_RETURN (
        ERROR, !access (file_path, R_OK), XCAM_RETURN_ERROR_PARAM,
        "cannot access intrinsic file %s", file_path);

    File file_reader;
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
CalibrationParser::parse_extrinsic_file (const char *file_path, ExtrinsicParameter &extrinsic_param)
{
    XCAM_FAIL_RETURN (
        ERROR, !access (file_path, R_OK), XCAM_RETURN_ERROR_PARAM,
        "cannot access extrinsic file %s", file_path);

    File file_reader;
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

XCamReturn
CalibrationParser::parse_calib_file (const char *file_path, std::vector<CalibrationInfo> &calib_info, int32_t camera_count)
{
    XCAM_FAIL_RETURN (
        ERROR, !access (file_path, R_OK), XCAM_RETURN_ERROR_PARAM,
        "cannot access extrinsic file %s", file_path);

    File file_reader;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    std::vector<char> context;
    size_t file_size = 0;

    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.open (file_path, "r")), ret,
        "open calibration file(%s) failed.", file_path);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.get_file_size (file_size)), ret,
        "read calibration file(%s) failed to get file size.", file_path);
    context.resize (file_size + 1);
    XCAM_FAIL_RETURN (
        WARNING, xcam_ret_is_ok (ret = file_reader.read_file (&context[0], file_size)), ret,
        "read calibration file(%s) failed, file size:%d.", file_path, (int)file_size);
    file_reader.close ();
    context[file_size] = '\0';

    return parse_calib_param (&context[0], calib_info, camera_count);
}

XCamReturn
CalibrationParser::parse_calib_param (char *file_body, std::vector<CalibrationInfo> &calib_info, int32_t camera_count)
{
    char *line_str = NULL;
    char *line_endptr = NULL;
    char *tok_str = NULL;
    char *tok_endptr = NULL;
    static const char *line_tokens = "\r\n";
    static const char *str_tokens = " \t";
    int32_t index = -1;

    do {
        if (NULL == line_endptr && -1 == index) {
            line_str = strtok_r (file_body, line_tokens, &line_endptr);
        } else {
            line_str = strtok_r (NULL, line_tokens, &line_endptr);
            if (NULL == line_str || index >= camera_count) {
                break;
            }
        }

        tok_str = strtok_r (line_str, str_tokens, &tok_endptr);
        XCAM_LOG_DEBUG ("Parse Attribute: %s", tok_str);
        CHECK_PARAM (tok_str);

        if (!strncmp (tok_str, calib_attribute[CalibCameraId], strnlen(tok_str, 10))) {
            if (++index >= camera_count) {
                break;
            }
            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            calib_info[index].camera_id = strtol (tok_str, NULL, 10);
            XCAM_LOG_DEBUG ("   Value: %d", calib_info[index].camera_id);
        }
        CHECK_PARAM (tok_str);

        if (!strncmp (tok_str, calib_attribute[CalibCameraMatrix], strnlen(tok_str, 10))) {
            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            XCAM_LOG_DEBUG ("   Value: %s", tok_str);
            calib_info[index].intrinsic.fx = strtof (tok_str, NULL);
            calib_info[index].intrinsic.fy = strtof (tok_str, NULL);

            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            XCAM_LOG_DEBUG ("   Value: %s", tok_str);
            calib_info[index].intrinsic.cx = strtof (tok_str, NULL);

            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            XCAM_LOG_DEBUG ("   Value: %s", tok_str);
            calib_info[index].intrinsic.cy = strtof (tok_str, NULL);

            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            XCAM_LOG_DEBUG ("   Value: %s", tok_str);
            calib_info[index].intrinsic.skew = strtof (tok_str, NULL);
        }
        CHECK_PARAM (tok_str);

        if (!strncmp (tok_str, calib_attribute[CalibRotationMatrix], strnlen(tok_str, 10))) {
            Mat3f rotation;
            uint32_t i = 0;
            while (NULL != (tok_str = strtok_r (NULL, str_tokens, &tok_endptr))) {
                rotation (i / 3, i % 3) = strtof (tok_str, NULL);
                XCAM_LOG_DEBUG ("   Value: %s", tok_str);
                i++;
            }
            Quaternion<float> quat = create_quaternion (rotation);

            //Pitch->X axis, Yaw->Y axis, Roll->Z axis
            //Measured in radians
            Vec3f euler = quat.euler_angles ();
            calib_info[index].extrinsic.pitch = RADIANS_2_DEGREE (euler[0]);
            calib_info[index].extrinsic.yaw = RADIANS_2_DEGREE (euler[1]);
            calib_info[index].extrinsic.roll = RADIANS_2_DEGREE (euler[2]);
        }
        CHECK_PARAM (tok_str);

        if (!strncmp (tok_str, calib_attribute[CalibTranslationMatrix], strnlen(tok_str, 10))) {
            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            XCAM_LOG_DEBUG ("   Value: %s", tok_str);
            calib_info[index].extrinsic.trans_x = strtof (tok_str, NULL);

            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            XCAM_LOG_DEBUG ("   Value: %s", tok_str);
            calib_info[index].extrinsic.trans_y = strtof (tok_str, NULL);

            tok_str = strtok_r (NULL, str_tokens, &tok_endptr);
            CHECK_NULL (tok_str);
            XCAM_LOG_DEBUG ("   Value: %s", tok_str);
            calib_info[index].extrinsic.trans_z = strtof (tok_str, NULL);
        }
        CHECK_PARAM (tok_str);
    } while (true);

    return XCAM_RETURN_NO_ERROR;
}

#if HAVE_JSON
XCamReturn
CalibrationParser::parse_fisheye_camera_param (const char *file_path, FisheyeInfo *fisheye_info, int32_t camera_count)
{
    XCAM_LOG_DEBUG ("Parse camera calibration file: %s", file_path);

    if (NULL == file_path) {
        XCAM_LOG_ERROR ("invalide input file path !");
        return XCAM_RETURN_ERROR_PARAM;
    }
    std::ifstream calibFile (file_path);
    if (!calibFile.is_open ()) {
        XCAM_LOG_ERROR ("calibration file Not Found!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    try {
        json calib_params = json::parse (calibFile);

        auto const model = calib_params.find ("model");
        if (model != calib_params.end ()) {
            for ( int i = 0; i < 6; i++) {
                fisheye_info[i].cam_model = model->get<int>();
            }
            XCAM_LOG_DEBUG ("camera model=%d ", calib_params.find ("model")->get<int>());
        } else {
            XCAM_LOG_WARNING ("model Not Found");
        }

        auto const cameras = calib_params.find ("cameras");
        if (cameras == calib_params.end ()) {
            XCAM_LOG_ERROR ("cameras Not Found");
            return XCAM_RETURN_ERROR_PARAM;
        }

        auto const camera = cameras->find ("camera");
        if (camera == cameras->end ()) {
            XCAM_LOG_ERROR ("camera Not Found");
            return XCAM_RETURN_ERROR_PARAM;
        }

        int32_t cam_id = 0;
        for (json::iterator cam = camera->begin (); cam != camera->end (), cam_id < camera_count; cam++) {

            auto const cam_radius = cam->find ("radius");
            if (cam_radius != cam->end ()) {
                fisheye_info[cam_id].radius = cam_radius->get<float>();
            }

            auto const cam_cx = cam->find ("cx");
            if (cam_cx != cam->end ()) {
                fisheye_info[cam_id].intrinsic.cx = cam_cx->get<float>();
            }

            auto const cam_cy = cam->find ("cy");
            if (cam_cy != cam->end ()) {
                fisheye_info[cam_id].intrinsic.cy = cam_cy->get<float>();
            }

            auto const cam_w = cam->find ("w");
            if (cam_w != cam->end ()) {
                fisheye_info[cam_id].intrinsic.width = cam_w->get<int>();
            }

            auto const cam_h = cam->find ("h");
            if (cam_h != cam->end ()) {
                fisheye_info[cam_id].intrinsic.height = cam_h->get<int>();
            }

            auto const cam_skew = cam->find ("skew");
            if (cam_skew != cam->end ()) {
                fisheye_info[cam_id].intrinsic.skew = cam_skew->get<float>();
            }

            auto const cam_fx = cam->find ("fx");
            if (cam_fx != cam->end ()) {
                fisheye_info[cam_id].intrinsic.fx = cam_fx->get<float>();
            }

            auto const cam_fy = cam->find ("fy");
            if (cam_fy != cam->end ()) {
                fisheye_info[cam_id].intrinsic.fy = cam_fy->get<float>();
            }

            auto const cam_fov = cam->find ("fov");
            if (cam_fov != cam->end ()) {
                fisheye_info[cam_id].intrinsic.fov = cam_fov->get<float>();
            }

            auto const cam_flip = cam->find ("flip");
            if (cam_flip != cam->end ()) {
                fisheye_info[cam_id].intrinsic.flip = (strcasecmp (cam_flip->get<std::string>().c_str (), "true") == 0 ? true : false);
            }
            XCAM_LOG_DEBUG ("cam[%d]: flip=%d ", cam_id, fisheye_info[cam_id].intrinsic.flip);
            XCAM_LOG_DEBUG ("fx=%f ", fisheye_info[cam_id].intrinsic.fx);
            XCAM_LOG_DEBUG ("fy=%f ", fisheye_info[cam_id].intrinsic.fy);
            XCAM_LOG_DEBUG ("cx=%f ", fisheye_info[cam_id].intrinsic.cx);
            XCAM_LOG_DEBUG ("cy=%f ", fisheye_info[cam_id].intrinsic.cy);
            XCAM_LOG_DEBUG ("w=%d ", fisheye_info[cam_id].intrinsic.width);
            XCAM_LOG_DEBUG ("h=%d ", fisheye_info[cam_id].intrinsic.height);
            XCAM_LOG_DEBUG ("fov=%f ", fisheye_info[cam_id].intrinsic.fov);
            XCAM_LOG_DEBUG ("skew=%f ", fisheye_info[cam_id].intrinsic.skew);

            auto const cam_yaw = cam->find ("yaw");
            if (cam_yaw != cam->end ()) {
                fisheye_info[cam_id].extrinsic.yaw = cam_yaw->get<float>();
            }

            auto const cam_pitch = cam->find ("pitch");
            if (cam_pitch != cam->end ()) {
                fisheye_info[cam_id].extrinsic.pitch = cam_pitch->get<float>();
            }

            auto const cam_roll = cam->find ("roll");
            if (cam_roll != cam->end ()) {
                fisheye_info[cam_id].extrinsic.roll = cam_roll->get<float>();
            }

            auto const cam_k = cam->find ("K");
            if (cam_k != cam->end ()) {
                uint32_t i = 0;
                for (json::iterator k_mat = cam_k->begin (); k_mat != cam_k->end (); k_mat++, i++) {
                    XCAM_LOG_DEBUG ("k[%d]: %f ", i, k_mat->get<float>());
                }
            }

            auto const cam_d = cam->find ("D");
            if (cam_d != cam->end ()) {
                uint32_t i = 0;
                for (json::iterator d_vec = cam_d->begin (); d_vec != cam_d->end (), i < 4; d_vec++, i++) {
                    fisheye_info[cam_id].distort_coeff[i] = d_vec->get<float>();
                    XCAM_LOG_DEBUG ("d[%d]: %f ", i, d_vec->get<float>());
                }
            }

            auto const cam_r = cam->find ("R");
            if (cam_r != cam->end ()) {
                Mat3f rotation;
                uint32_t i = 0;
                for (json::iterator r_mat = cam_r->begin (); r_mat != cam_r->end (), i < 9; r_mat++, i++) {
                    rotation (i / 3, i % 3) = r_mat->get<float>();
                }
                Quaternion<float> quat = create_quaternion (rotation);
                //Pitch->X axis, Yaw->Y axis, Roll->Z axis
                //Measured in radians
                Vec3f euler = quat.euler_angles ();
                fisheye_info[cam_id].extrinsic.pitch = RADIANS_2_DEGREE (euler[0]);
                fisheye_info[cam_id].extrinsic.yaw = RADIANS_2_DEGREE (euler[1]);
                fisheye_info[cam_id].extrinsic.roll = RADIANS_2_DEGREE (euler[2]);
            }

            auto const cam_t = cam->find ("t");
            if (cam_t != cam->end ()) {
                uint32_t i = 0;
                Vec3f translation;
                for (json::iterator t_vec = cam_t->begin (); t_vec != cam_t->end (), i < 3; t_vec++, i++) {
                    translation[i] = t_vec->get<float>();
                    XCAM_LOG_DEBUG ("t[%d]: %f ", i, t_vec->get<float>());
                }
                fisheye_info[cam_id].extrinsic.trans_x = translation[0];
                fisheye_info[cam_id].extrinsic.trans_y = translation[1];
                fisheye_info[cam_id].extrinsic.trans_z = translation[2];
            }

            auto const cam_c = cam->find ("c");
            if (cam_c != cam->end ()) {
                uint32_t i = 0;
                for (json::iterator c_vec = cam_c->begin (); c_vec != cam_c->end (), i < 3; c_vec++, i++) {
                    fisheye_info[cam_id].c_coeff[i] = c_vec->get<float>();
                    XCAM_LOG_DEBUG ("c[%d]: %f ", i, c_vec->get<float>());
                }
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
