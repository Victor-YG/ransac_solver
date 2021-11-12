#pragma once

#include <vector>

#include "opencv4/opencv2/features2d.hpp"


// normalize a set of keypoints and return the 3x3 normalization matrix
cv::Mat Normalize(const std::vector<cv::Point2f>& keypoints, std::vector<cv::Point2f>& normalized_keypoints);

cv::Mat CalcEssentialMatrix(std::vector<cv::Point2f> kp_1, std::vector<cv::Point2f> kp_2);

void DecomposeEssentialMatrix(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t);