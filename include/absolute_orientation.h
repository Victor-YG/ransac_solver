#pragma once

#include "ransac_solver.h"

#include <vector>

#include <Eigen/Geometry>


typedef std::pair<Eigen::Vector3f, Eigen::Vector3f> PointPair;

Eigen::Vector3f AveragePointCloud(std::vector<Eigen::Vector3f> points);

Eigen::Matrix3f CrossCovariance(
    std::vector<Eigen::Vector3f> src, 
    std::vector<Eigen::Vector3f> dst);


class PointSetTransModel : public RANSAC::Model<PointPair, Eigen::Matrix4f>
{
public:
    unsigned int NumElementsRequired() override
    {
        return 4;
    };

    Eigen::Matrix4f Fit(
        const std::vector<PointPair>& elements, 
        const std::vector<float>& weights) override
    {
        unsigned int N = elements.size();
        
        std::vector<Eigen::Vector3f> src;
        std::vector<Eigen::Vector3f> dst;
        src.reserve(N);
        dst.reserve(N);

        // compute centroids of point clouds
        Eigen::Vector3f src_c = Eigen::Vector3f::Zero();
        Eigen::Vector3f dst_c = Eigen::Vector3f::Zero();

        for (int i = 0; i < N; i++)
        {
            src_c += elements[i].first;
            dst_c += elements[i].second;
        }

        src_c /= N;
        dst_c /= N;

        // demean of point clouds
        for (int i = 0; i < N; i++)
        {
            src.emplace_back(elements[i].first  - src_c);
            dst.emplace_back(elements[i].second - dst_c);
        }

        // compute cross covariance matrix
        Eigen::Matrix3f M = Eigen::Matrix3f::Zero();

        for (int i = 0; i < N; i++)
        {
            M(0, 0) += dst[i](0) * src[i](0);
            M(0, 1) += dst[i](0) * src[i](1);
            M(0, 2) += dst[i](0) * src[i](2);
            M(1, 0) += dst[i](1) * src[i](0);
            M(1, 1) += dst[i](1) * src[i](1);
            M(1, 2) += dst[i](1) * src[i](2);
            M(2, 0) += dst[i](2) * src[i](0);
            M(2, 1) += dst[i](2) * src[i](1);
            M(2, 2) += dst[i](2) * src[i](2);
        }

        // svd
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix U = svd.matrixU();
        Eigen::Matrix V = svd.matrixV();

        Eigen::Vector3f S = Eigen::Vector3f::Ones();
        if (U.determinant() * V.determinant() < 0) S(2) = -1;

        // compute transformation
        m_rotation = U * S.asDiagonal() * V.transpose();
        m_translation = dst_c - m_rotation * src_c;

        m_trans = Eigen::Matrix4f::Identity();
        m_trans(0, 0) = m_rotation(0, 0);
        m_trans(0, 1) = m_rotation(0, 1);
        m_trans(0, 2) = m_rotation(0, 2);
        m_trans(1, 0) = m_rotation(1, 0);
        m_trans(1, 1) = m_rotation(1, 1);
        m_trans(1, 2) = m_rotation(1, 2);
        m_trans(2, 0) = m_rotation(2, 0);
        m_trans(2, 1) = m_rotation(2, 1);
        m_trans(2, 2) = m_rotation(2, 2);
        m_trans(0, 3) = m_translation(0);
        m_trans(1, 3) = m_translation(1);
        m_trans(2, 3) = m_translation(2);
        
        return m_trans;
    }

    bool IsInlier(const PointPair& element, float& loss) override
    {
        Eigen::Vector3f src = element.first;
        Eigen::Vector3f dst = element.second;

        Eigen::Vector3f src_trans = m_rotation * src + m_translation;
        Eigen::Vector3f diff = dst - src_trans;

        float dist = 0.0;
        dist += diff(0) * diff(0);
        dist += diff(1) * diff(1);
        dist += diff(2) * diff(2);
        dist = sqrt(dist);
        loss = dist; // L2 distance

        return (dist < 1.0);
    }

private:
    // variables
    Eigen::Matrix4f m_trans;
    Eigen::Matrix3f m_rotation;
    Eigen::Vector3f m_translation;
};
