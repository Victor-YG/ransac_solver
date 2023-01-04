#include "reconstruction.h"

#include "essential_matrix.h"

#include <stdlib.h>
#include <iostream>


// #define DEBUG

// cv::Mat Normalize(const std::vector<cv::Point2f>& keypoints, std::vector<cv::Point2f>& normalized_keypoints)
// {
//     unsigned int N = keypoints.size();
//     normalized_keypoints.resize(N);
//
//     for (int i = 0; i < N; i++)
//     {
//         normalized_keypoints[i].x = keypoints[i].x;
//         normalized_keypoints[i].y = keypoints[i].y;
//     }
//
//     return cv::Mat::eye(3, 3, CV_32F);
// }

cv::Mat Normalize(const std::vector<cv::Point2f>& keypoints, std::vector<cv::Point2f>& normalized_keypoints)
{
    unsigned int N = keypoints.size();
    normalized_keypoints.resize(N);

    // compute mean
    float mean_x = 0;
    float mean_y = 0;

    for (int i = 0; i < N; i++)
    {
        mean_x += keypoints[i].x;
        mean_y += keypoints[i].y;
    }

    mean_x = mean_x / N;
    mean_y = mean_y / N;

    // compute average deviation
    float total_dev_x = 0;
    float total_dev_y = 0;

    for (int i = 0; i < N; i++)
    {
        // zero center all keypoints
        normalized_keypoints[i].x = keypoints[i].x - mean_x;
        normalized_keypoints[i].y = keypoints[i].y - mean_y;
        // accumuate deviation
        total_dev_x += abs(normalized_keypoints[i].x);
        total_dev_y += abs(normalized_keypoints[i].y);
    }

    float scale_x = N / total_dev_x;
    float scale_y = N / total_dev_y;

    // scale all keypoints
    for (int i = 0; i < N; i++)
    {
        normalized_keypoints[i].x *= scale_x;
        normalized_keypoints[i].y *= scale_y;
    }

    // update normalization matrix
    cv::Mat normalization = cv::Mat::eye(3, 3, CV_32F);
    normalization.at<float>(0, 0) =  scale_x;
    normalization.at<float>(1, 1) =  scale_y;
    normalization.at<float>(0, 2) = -mean_x * scale_x;
    normalization.at<float>(1, 2) = -mean_y * scale_y;

    #ifdef DEBUG
    std::cout << "[DBUG]: reconstruction::Normalized(): number N = " << N << std::endl;
    std::cout << "[DBUG]: reconstruction::Normalized(): mean_x = " << mean_x << ", mean_y = " << mean_y << std::endl;
    std::cout << "[DBUG]: reconstruction::Normalized(): scale_x = " << scale_x << ", scale_y = " << scale_y << std::endl;
    #endif

    return normalization;
}

cv::Mat CalcEssentialMatrix(std::vector<cv::Point2f> kp_1, std::vector<cv::Point2f> kp_2)
{
    if (kp_1.size() != kp_2.size() || kp_1.size() == 0)
    {
        std::cout << "[EROR]: size mismatch of keypoint lists in CalcEssentialMatrix()." << std::endl;
    }

    // normalization
    std::vector<cv::Point2f> normalized_kp_1;
    std::vector<cv::Point2f> normalized_kp_2;

    cv::Mat N1 = Normalize(kp_1, normalized_kp_1);
    cv::Mat N2 = Normalize(kp_2, normalized_kp_2);

    // get point correspondences
    std::vector<PointPair> point_pairs;
    for (int i = 0; i < kp_1.size(); i++)
    {
        point_pairs.emplace_back(std::make_pair(normalized_kp_1[i], normalized_kp_2[i]));
    }

    // compute essential matrix
    EssentialMatrix model;
    RANSAC::Solver<PointPair, cv::Mat> solver(&model);
    cv::Mat E_hat = solver.Solve(point_pairs);
    cv::Mat E = N2.t() * E_hat * N1;

    return E;

    #ifdef DEBUG
    std::cout << "[DBUG]: reconstruction::CalcEssentialMatrix(): N1 = " << std::endl;
    std::cout << N1 << std::endl;

    std::cout << "[DBUG]: reconstruction::CalcEssentialMatrix(): after normalization = " << std::endl;
    std::cout << normalized_kp_1[0] << std::endl;

    std::cout << "[DBUG]: reconstruction::CalcEssentialMatrix(): essential matrix = " << std::endl;
    std::cout << mat << std::endl;

    std::cout << "[DBUG]: reconstruction::CalcEssentialMatrix(): final essential matrix = " << std::endl;
    std::cout << E << std::endl;
    #endif
}

void DecomposeEssentialMatrix(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t)
{
    cv::Mat U, w, Vt;
    cv::SVD::compute(E, w, U, Vt, cv::SVD::FULL_UV);

    U.col(2).copyTo(t);
    t = t / cv::norm(t);

    cv::Mat W = cv::Mat::zeros(3, 3, CV_32F);
    W.at<float>(0, 1) = -1.0;
    W.at<float>(1, 0) =  1.0;
    W.at<float>(2, 2) =  1.0;

    R1 = U * W     * Vt;
    R2 = U * W.t() * Vt;

    if (cv::determinant(R1) < 0)
        R1 = -R1;
    if (cv::determinant(R2) < 0)
        R2 = -R2;

    #ifdef DEBUG
    std::cout << "[DBUG]: input essential = " << std::endl;
    std::cout << E << std::endl;

    std::cout << "[DBUG]: decomposed R1 = " << std::endl;
    std::cout << R1 << std::endl;

    std::cout << "[DBUG]: decomposed R2 = " << std::endl;
    std::cout << R2 << std::endl;

    std::cout << "[DBUG]: decomposed t = " << std::endl;
    std::cout << t << std::endl;
    #endif
}
