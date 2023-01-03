#pragma once

#include "ransac_solver.h"

#include <Eigen/Geometry>


typedef std::pair<Eigen::Vector3f, Eigen::Vector3f> PointPair;

class PointSetTransModel : public RANSAC::Model<PointPair, Eigen::Matrix4f>
{
public:
    unsigned int NumElementsRequired() override
    {
        return 3;
    }

    void SetModelParams(const Eigen::Matrix4f& params) override
    {
        m_trans = params;
    }

    Eigen::Matrix4f Fit(
        const std::vector<PointPair>& elements,
        const std::vector<float>& weights) override
    {
        unsigned int N = elements.size();

        Eigen::MatrixXf src(3, N);
        Eigen::MatrixXf dst(3, N);

        for (int i = 0; i < N; i++)
        {
            src(0, i) = elements[i].first(0);
            src(1, i) = elements[i].first(1);
            src(2, i) = elements[i].first(2);
            dst(0, i) = elements[i].second(0);
            dst(1, i) = elements[i].second(1);
            dst(2, i) = elements[i].second(2);
        }

        m_trans = Eigen::umeyama(src, dst, false);
        return m_trans;
    }

    bool IsInlier(const PointPair& element, float& loss) override
    {
        Eigen::Vector3f src = element.first;
        Eigen::Vector3f dst = element.second;

        Eigen::Vector4f src_h;
        src_h(0) = src(0);
        src_h(1) = src(1);
        src_h(2) = src(2);
        src_h(3) = 1.0;

        Eigen::Vector4f src_trans = m_trans * src_h;

        float dist = 0.0;
        dist += (dst(0) - src_trans(0)) * (dst(0) - src_trans(0));
        dist += (dst(1) - src_trans(1)) * (dst(1) - src_trans(1));
        dist += (dst(2) - src_trans(2)) * (dst(2) - src_trans(2));
        dist = sqrt(dist);

        if (dist < 1.0)
        {
            loss = 0;
            return true;
        }
        loss = 0;
        return false;
    }

private:
    Eigen::Matrix4f m_trans;
};
