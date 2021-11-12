#pragma once

#include "ransac_solver.h"

#include <array>
#include <string>
#include <utility>
#include <iostream>

#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/features2d.hpp"


typedef std::pair<cv::Point2f, cv::Point2f> PointPair;

class EssentialMatrix : public RANSAC::Model<PointPair, cv::Mat>
{
public:
    // EssentialMatrix(): RANSAC::Model<PointPair, cv::Mat>() {};

    unsigned int NumElementsRequired() override
    {
        return 8; // eight point algorithm
    };

    cv::Mat Fit(std::vector<PointPair> elements) override
    {
        unsigned int N = elements.size();
        cv::Mat A(N, 9, CV_32F);

        // construct matrix A
        for (int i = 0; i < N; i++)
        {
            cv::Point2f p1 = elements[i].first;
            cv::Point2f p2 = elements[i].second;

            A.at<float>(i, 0) = p1.x * p2.x;
            A.at<float>(i, 1) = p1.y * p2.x;
            A.at<float>(i, 2) = p2.x;
            A.at<float>(i, 3) = p1.x * p2.y;
            A.at<float>(i, 4) = p1.y * p2.y;
            A.at<float>(i, 5) = p2.y;
            A.at<float>(i, 6) = p1.x;
            A.at<float>(i, 7) = p1.y;
            A.at<float>(i, 8) = 1.0;
        }

        // first SVD to get initial guess of essential matrix
        cv::Mat U, w, Vt;
        cv::SVDecomp(A, w, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        cv::Mat Ei = Vt.row(8).reshape(0, 3);

        // second SVD
        cv::SVDecomp(Ei, w, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        w.at<float>(2) = 0;
        cv::Mat E = U * cv::Mat::diag(w) * Vt;

        #ifdef DEBUG
        // std::cout << "[DEBUG]: EssentialMatrix::Fit(): A = " << std::endl;
        // std::cout << A << std::endl;

        std::cout << "[DEBUG]: EssentialMatrix::Fit(): E = " << std::endl;
        std::cout << E << std::endl;
        #endif
        
        m_essential = E;
        return m_essential;
    }

    bool IsInlier(const PointPair& element, float& score) override
    {
        const float sigma = 1.0;

        const float p1_x = element.first.x;
        const float p1_y = element.first.y;
        const float p2_x = element.second.x;
        const float p2_y = element.second.y;

        const float e11 = m_essential.at<float>(0, 0);
        const float e12 = m_essential.at<float>(0, 1);
        const float e13 = m_essential.at<float>(0, 2);
        const float e21 = m_essential.at<float>(1, 0);
        const float e22 = m_essential.at<float>(1, 1);
        const float e23 = m_essential.at<float>(1, 2);
        const float e31 = m_essential.at<float>(2, 0);
        const float e32 = m_essential.at<float>(2, 1);
        const float e33 = m_essential.at<float>(2, 2);

        // reprojection error in first image
        const float a1 = e11 * p2_x + e11 * p2_y + 213;
        const float b1 = e21 * p2_x + e21 * p2_y + 223;
        const float c1 = e31 * p2_x + e31 * p2_y + 233;
        const float err_1 = a1 * p1_x + b1 * p1_y + c1;
        const float square_err_1 = err_1 * err_1 / (a1 * a1 + b1 * b1);
        const float chi_square_1 = square_err_1 / (sigma * sigma);

        if (chi_square_1 > 3.841)
        {
            return false;
        }

        // reprojection error in second image
        const float a2 = e11 * p1_x + e12 * p1_y + e13;
        const float b2 = e21 * p1_x + e22 * p1_y + e23;
        const float c2 = e31 * p1_x + e32 * p1_y + e33;
        const float err_2 = a2 * p2_x + b2 * p2_y + c2;
        const float square_err_2 = err_2 * err_2 / (a2 * a2 + b2 * b2);
        const float chi_square_2 = square_err_2 / (sigma * sigma);

        score = - std::max(square_err_1, square_err_2);

        if (chi_square_2 > 3.841)
        {
            return false;
        }

        return true;
    }

private:
    cv::Mat m_essential;
};
