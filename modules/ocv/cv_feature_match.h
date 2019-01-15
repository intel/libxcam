/*
 * cv_feature_match.h - optical flow feature match
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_CV_FEATURE_MATCH_H
#define XCAM_CV_FEATURE_MATCH_H

#include <video_buffer.h>
#include <interface/feature_match.h>
#include "cv_utils.h"

namespace XCam {

class CVFeatureMatch
    : public FeatureMatch
{
public:
    enum BufId {
        BufId0    = 0,
        BufId1,
        BufIdMax
    };

public:
    explicit CVFeatureMatch ();
    virtual ~CVFeatureMatch ();

    virtual void optical_flow_feature_match (
        const SmartPtr<VideoBuffer> &left_buf, const SmartPtr<VideoBuffer> &right_buf,
        Rect &left_img_crop, Rect &right_img_crop, int dst_width = 0);

    void set_cl_buf_mem (void *mem, BufId id);

protected:
    bool get_crop_image_umat (const SmartPtr<VideoBuffer> &buffer, const Rect &crop_rect, cv::UMat &img, BufId id);

    void add_detected_data (cv::Mat image, cv::Ptr<cv::Feature2D> detector, std::vector<cv::Point2f> &corners);
    void get_valid_offsets (std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
                            std::vector<uchar> &status, std::vector<float> &error,
                            std::vector<float> &offsets, float &sum, int &count,
                            cv::Mat debug_img, cv::Size &img0_size);

    void calc_of_match (cv::Mat image0, cv::Mat image1,
                        std::vector<cv::Point2f> &corner0, std::vector<cv::Point2f> &corner1,
                        std::vector<uchar> &status, std::vector<float> &error,
                        int &last_count, float &last_mean_offset, float &out_x_offset);

    void detect_and_match (cv::Mat img_left, cv::Mat img_right, Rect &crop_left, Rect &crop_right,
                           int &valid_count, float &mean_offset, float &x_offset, int dst_width);

    void adjust_stitch_area (int dst_width, float &x_offset, Rect &stitch0, Rect &stitch1);

    void debug_write_image ( const SmartPtr<VideoBuffer> &buf, const Rect &rect, char *img_name,
                             char *frame_str, char *fm_idx_str);

private:
    XCAM_DEAD_COPY (CVFeatureMatch);

    void        *_cl_buf_mem[BufIdMax];
};

}

#endif // XCAM_CV_FEATURE_MATCH_H
