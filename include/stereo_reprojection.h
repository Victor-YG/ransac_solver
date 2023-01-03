#pragma once

#include "ransac_solver.h"

#include <Eigen/Geometry>


typedef std::pair<Eigen::Vector4f, Eigen::Vector4f> ObservationPair;


static inline Eigen::Vector3f StereoReprojection(const std::array<float, 16>& reprojection, Eigen::Vector4f stereo_obs)
{
    float u_l = stereo_obs(0);
    float v_l = stereo_obs(1);
    float u_r = stereo_obs(2);
    float v_r = stereo_obs(3);
    float d = u_l - u_r;

    float fw_inv = 1.0 / (d * reprojection[14]) + reprojection[15];
    float x = fw_inv * (u_l * reprojection[0] + reprojection[3]);
    float y = fw_inv * (v_l * reprojection[5] + reprojection[7]);
    float z = reprojection[11] * fw_inv;

    return Eigen::Vector3f(x, y, z);
}

class StereoTransModel : public RANSAC::Model<ObservationPair, Eigen::Matrix4f>
{
public:
    void SetCameraParams(cv::Mat projection_l, cv::Mat projection_r)
    {
        m_projection_l = projection_l;
        m_projection_r = projection_r;

        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                m_projection_l_eigen(r, c) = m_projection_l.at<float>(r, c);
                m_projection_r_eigen(r, c) = m_projection_r.at<float>(r, c);
            }
        }
    }

    unsigned int NumElementsRequired() override
    {
        return 3;
    }

    void SetModelParams(const Eigen::Matrix4f& params) override
    {
        m_trans = params;
    }

    Eigen::Matrix4f Fit(
        const std::vector<ObservationPair>& elements,
        const std::vector<float>& weights) override
    {
        unsigned int N = elements.size();
        Eigen::MatrixXf src(3, N);
        Eigen::MatrixXf dst(3, N);

        // prepare vector of keypoints
        std::vector<cv::Point2f> keypoints_l_1, keypoints_r_1;
        std::vector<cv::Point2f> keypoints_l_2, keypoints_r_2;
        for (int i = 0; i < N; i++)
        {
            Eigen::Vector4f obs_1 = elements[i].first;
            Eigen::Vector4f obs_2 = elements[i].second;
            keypoints_l_1.emplace_back(cv::Point2f(obs_1(0), obs_1(1)));
            keypoints_r_1.emplace_back(cv::Point2f(obs_1(2), obs_1(3)));
            keypoints_l_2.emplace_back(cv::Point2f(obs_2(0), obs_2(1)));
            keypoints_r_2.emplace_back(cv::Point2f(obs_2(2), obs_2(3)));
        }

        // triangulate
        cv::Mat points_homo_1, points_homo_2;
        cv::triangulatePoints(m_projection_l, m_projection_r, keypoints_l_1, keypoints_r_1, points_homo_1);
        cv::triangulatePoints(m_projection_l, m_projection_r, keypoints_l_2, keypoints_r_2, points_homo_2);

        // prepare matrix
        for (int i = 0; i < N; i++)
        {
            src(0, i) = points_homo_1.at<float>(0, i) / points_homo_1.at<float>(3, i);
            src(1, i) = points_homo_1.at<float>(1, i) / points_homo_1.at<float>(3, i);
            src(2, i) = points_homo_1.at<float>(2, i) / points_homo_1.at<float>(3, i);
            dst(0, i) = points_homo_2.at<float>(0, i) / points_homo_2.at<float>(3, i);
            dst(1, i) = points_homo_2.at<float>(1, i) / points_homo_2.at<float>(3, i);
            dst(2, i) = points_homo_2.at<float>(2, i) / points_homo_2.at<float>(3, i);
        }

        m_trans = Eigen::umeyama(src, dst, false);
        return m_trans;
    }

    bool IsInlier(const ObservationPair& element, float& loss) override
    {
        cv::Mat point_homo;
        Eigen::Vector4f obs_src = element.first;
        Eigen::Vector4f obs_dst = element.second;
        std::vector<cv::Point2f> kp_1, kp_2;
        kp_1.emplace_back(cv::Point2f(obs_src(0), obs_src(1)));
        kp_2.emplace_back(cv::Point2f(obs_src(2), obs_src(3)));

        cv::triangulatePoints(m_projection_l, m_projection_r, kp_1, kp_2, point_homo);

        Eigen::Vector4f src_h = Eigen::Vector4f::Zero();
        src_h(0) = point_homo.at<float>(0) / point_homo.at<float>(3);
        src_h(1) = point_homo.at<float>(1) / point_homo.at<float>(3);
        src_h(2) = point_homo.at<float>(2) / point_homo.at<float>(3);
        src_h(3) = 1.0;

        Eigen::Vector4f src_trans = m_trans * src_h;
        Eigen::Vector3f obs_trans_l = m_projection_l_eigen * src_trans;
        Eigen::Vector3f obs_trans_r = m_projection_r_eigen * src_trans;
        obs_trans_l /= obs_trans_l(2);
        obs_trans_r /= obs_trans_r(2);

        float dist = 0.0;
        dist += (obs_dst(0) - obs_trans_l(0)) * (obs_dst(0) - obs_trans_l(0));
        dist += (obs_dst(1) - obs_trans_l(1)) * (obs_dst(1) - obs_trans_l(1));
        dist += (obs_dst(2) - obs_trans_r(0)) * (obs_dst(2) - obs_trans_r(0));
        dist += (obs_dst(3) - obs_trans_r(1)) * (obs_dst(3) - obs_trans_r(1));
        dist = sqrt(dist);

        if (dist < 3.0) // RMSE is less than 3 pixel
        {
            loss = dist;
            return true;
        }
        loss = dist;
        return false;
    }

private:
    cv::Mat m_projection_l;
    cv::Mat m_projection_r;
    Eigen::Matrix<float, 3, 4> m_projection_l_eigen;
    Eigen::Matrix<float, 3, 4> m_projection_r_eigen;

    Eigen::Matrix4f m_trans;
};

class RectifiedStereoTransModel : public RANSAC::Model<ObservationPair, Eigen::Matrix4f>
{
public:
    void SetCameraParams(float fx, float fy, float cx, float cy, float b)
    {
        memset(&m_reprojection[0], 0, 16 * sizeof(float));
        m_projection_l = Eigen::Matrix<float, 3, 4>::Zero();
        m_projection_r = Eigen::Matrix<float, 3, 4>::Zero();

        m_reprojection[ 0] = 1.0;
        m_reprojection[ 1] = 0.0;
        m_reprojection[ 2] = 0.0;
        m_reprojection[ 3] = - cx;
        m_reprojection[ 4] = 0.0;
        m_reprojection[ 5] = 1.0;
        m_reprojection[ 6] = 0.0;
        m_reprojection[ 7] = - cy;
        m_reprojection[ 8] = 0.0;
        m_reprojection[ 9] = 0.0;
        m_reprojection[10] = 0.0;
        m_reprojection[11] = fx;
        m_reprojection[12] = 0.0;
        m_reprojection[13] = 0.0;
        m_reprojection[14] = 1.0 / b;
        m_reprojection[15] = 0.0;

        m_projection_l(0, 0) = fx;
        m_projection_l(0, 2) = cx;
        m_projection_l(0, 3) = 0.0;
        m_projection_l(1, 1) = fx;
        m_projection_l(1, 2) = cy;
        m_projection_l(1, 3) = 0.0;
        m_projection_l(2, 2) = 1.0;
        m_projection_l(2, 3) = 0.0;

        m_projection_r(0, 0) = fx;
        m_projection_r(0, 2) = cx;
        m_projection_r(0, 3) = - fx * 94.902;
        m_projection_r(1, 1) = fx;
        m_projection_r(1, 2) = cy;
        m_projection_r(1, 3) = 0.0;
        m_projection_r(2, 2) = 1.0;
        m_projection_r(2, 3) = 0.0;
    }

    unsigned int NumElementsRequired() override
    {
        return 3;
    }

    void SetModelParams(const Eigen::Matrix4f& params)
    {
        m_trans = params;
    }

    Eigen::Matrix4f Fit(
        const std::vector<ObservationPair>& elements,
        const std::vector<float>& weights) override
    {
        unsigned int N = elements.size();
        Eigen::MatrixXf src(3, N);
        Eigen::MatrixXf dst(3, N);

        // reproject stereo observations to 3d points
        for (int i = 0; i < N; i++)
        {
            Eigen::Vector3f p1 = StereoReprojection(m_reprojection, elements[i].first);
            Eigen::Vector3f p2 = StereoReprojection(m_reprojection, elements[i].second);

            src(0, i) = p1(0);
            src(1, i) = p1(1);
            src(2, i) = p1(2);
            dst(0, i) = p2(0);
            dst(1, i) = p2(1);
            dst(2, i) = p2(2);
        }

        m_trans = Eigen::umeyama(src, dst, false);
        return m_trans;
    }

    bool IsInlier(const ObservationPair& element, float& loss) override
    {
        Eigen::Vector4f obs_src = element.first;
        Eigen::Vector4f obs_dst = element.second;
        Eigen::Vector3f src = StereoReprojection(m_reprojection, obs_src);

        Eigen::Vector4f src_h;
        src_h(0) = src(0);
        src_h(1) = src(1);
        src_h(2) = src(2);
        src_h(3) = 1.0;

        Eigen::Vector4f src_trans = m_trans * src_h;
        Eigen::Vector3f obs_trans_l = m_projection_l * src_trans;
        Eigen::Vector3f obs_trans_r = m_projection_r * src_trans;
        obs_trans_l /= obs_trans_l(2);
        obs_trans_r /= obs_trans_r(2);

        float dist = 0.0;
        dist += (obs_dst(0) - obs_trans_l(0)) * (obs_dst(0) - obs_trans_l(0));
        dist += (obs_dst(1) - obs_trans_l(1)) * (obs_dst(1) - obs_trans_l(1));
        dist += (obs_dst(2) - obs_trans_r(0)) * (obs_dst(2) - obs_trans_r(0));
        dist += (obs_dst(3) - obs_trans_r(1)) * (obs_dst(3) - obs_trans_r(1));
        dist = sqrt(dist);

        if (dist < 5.0) // RMSE is less than 5 pixel
        {
            loss = dist;
            return true;
        }
        loss = dist;
        return false;
    }

private:
    std::array<float, 16> m_reprojection;
    Eigen::Matrix<float, 3, 4> m_projection_l;
    Eigen::Matrix<float, 3, 4> m_projection_r;

    Eigen::Matrix4f m_trans;
};
