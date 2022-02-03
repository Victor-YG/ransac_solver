#include "ransac_solver.h"
#include "reconstruction.h"
#include "essential_matrix.h"

#include <array>
#include <string>
#include <utility>
#include <fstream>
#include <iostream>

#include <Eigen/SVD>
#include <Eigen/Geometry>

#include "gflags/gflags.h"

#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/features2d.hpp"


// input variables
DEFINE_string(img_1, "", "First image.");
DEFINE_string(img_2, "", "Second image.");
DEFINE_string(camera, "", "Camera intrinsic.");


void LoadCameraParams(const std::string& file_path, float& fx, float& fy, float& cx, float& cy, float& k1, float& k2, float& p1, float& p2, float& k3);

int main(int argc, char** argv)
{
    // read inputs
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_img_1.empty())
    {
        std::cerr << "[FAIL]: Please provide an image file name using -img_1.\n";
        return -1;
    }

    if (FLAGS_img_2.empty()) {
        std::cerr << "[FAIL]: Please provide another image using -img_2.\n";
        return -1;
    }

    if (FLAGS_camera.empty())
    {
        std::cerr << "[FAIL]: Please provide camera intrinsic file using -camera. See ./examples/example_camera.txt for reference.\n";
        return -1;
    }

    std::string img_path_1 = FLAGS_img_1;
    std::string img_path_2 = FLAGS_img_2;

    cv::Mat img_1 = cv::imread(img_path_1, cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = cv::imread(img_path_2, cv::IMREAD_GRAYSCALE);
    int width  = img_1.cols;
    int height = img_1.rows;
    std::cout << "[INFO]: image size: " << width << " x " << height << std::endl;

    // ********************
    // * feature matching *
    // ********************    
    // initialize
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> describer = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;

    // extract ORB features
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    describer->compute(img_1, keypoints_1, descriptors_1);
    describer->compute(img_2, keypoints_2, descriptors_2);

    cv::Mat img_1_feature;
    cv::Mat img_2_feature;
    cv::drawKeypoints(img_1, keypoints_1, img_1_feature);
    cv::drawKeypoints(img_2, keypoints_2, img_2_feature);

    // matching
    std::vector<std::vector<cv::DMatch>> matched_feature_list;
    matcher->knnMatch(descriptors_1, descriptors_2, matched_feature_list, 2);

    // conduct Lowe's ratio test
    std::vector<cv::DMatch> final_matches;
    const float dist_thres = 30.0;
    const float ratio_thres = 0.5;

    for (int i = 0; i < matched_feature_list.size(); i++)
    {
        float d1 = matched_feature_list[i][0].distance;
        float d2 = matched_feature_list[i][1].distance;

        if (d1 < dist_thres && d1 / d2 < ratio_thres)
        {
            final_matches.emplace_back(matched_feature_list[i][0]);
        }
    }

    cv::Mat img_matched_features;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, final_matches, img_matched_features);
    cv::imshow("Final", img_matched_features);
    std::cout << "[INFO]: Found " << final_matches.size() << " pairs of good features." << std::endl;

    // ***************************
    // * undistort and normalize *
    // ***************************
    // load camera parameters
    std::string cam_file_path = FLAGS_camera;
    float fx, fy, cx, cy, k1, k2, p1, p2, k3;
    LoadCameraParams(cam_file_path, fx, fy, cx, cy, k1, k2, p1, p2, k3);
    cv::Mat_<float> cam_matrix(3, 3);
    cam_matrix << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    cv::Mat_<float> distortion_coef(1, 5);
    distortion_coef << k1, k2, p1, p2, k3;

    // undistort points
    cv::Mat_<cv::Point2f> list_kp_1(1, final_matches.size());
    cv::Mat_<cv::Point2f> list_kp_2(1, final_matches.size());
    for (int i = 0; i < final_matches.size(); i++)
    {
        int idx_1 = final_matches[i].queryIdx;
        int idx_2 = final_matches[i].trainIdx;
        cv::KeyPoint kp_1 = keypoints_1[idx_1];
        cv::KeyPoint kp_2 = keypoints_2[idx_2];
        list_kp_1(i) = kp_1.pt;
        list_kp_2(i) = kp_2.pt;
    }

    cv::Mat_<cv::Point2f> undistorted_kp_1;
    cv::Mat_<cv::Point2f> undistorted_kp_2;
    cv::undistortPoints(list_kp_1, undistorted_kp_1, cam_matrix, distortion_coef, cv::noArray(), cv::noArray());
    cv::undistortPoints(list_kp_2, undistorted_kp_2, cam_matrix, distortion_coef, cv::noArray(), cv::noArray());    

    // compute essential matrix
    cv::Mat essential = CalcEssentialMatrix(undistorted_kp_1, undistorted_kp_2);

    std::cout << "[INFO]: final essential matrix = " << std::endl;
    std::cout << essential << std::endl;

    // decompose into translation (baseline vector) and rotation
    cv::Mat R1, R2, t;
    DecomposeEssentialMatrix(essential, R1, R2, t);

    std::cout << "[INFO]: R1 = " << std::endl;
    std::cout << R1 << std::endl;
    std::cout << "[INFO]: R2 = " << std::endl;
    std::cout << R2 << std::endl;
    std::cout << "[INFO]: t = " << std::endl;
    std::cout << t << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

void LoadCameraParams(const std::string& file_path, float& fx, float& fy, float& cx, float& cy, float& k1, float& k2, float& p1, float& p2, float& k3)
{
    std::ifstream stream(file_path, std::ifstream::in);

    if (!stream.is_open())
    {
        std::cout << "[FAIL]: Failed to open camera file." << std::endl;
        return;
    }

    char char_arr[512];
    while (stream.getline(char_arr, 512))
    {
        if (stream.bad() || stream.eof())
            break;

        std::string line(char_arr);
        int idx = line.find_first_of("=");
        std::string key = line.substr(0, idx);
        std::string value = line.substr(idx + 1, line.length());

        if      (key == "fx") fx = stod(value);
        else if (key == "fy") fy = stod(value);
        else if (key == "cx") cx = stod(value);
        else if (key == "cy") cy = stod(value);
        else if (key == "k1") k1 = stod(value);
        else if (key == "k2") k2 = stod(value);
        else if (key == "p1") p1 = stod(value);
        else if (key == "p2") p2 = stod(value);
        else if (key == "k3") k3 = stod(value);
        else 
            std::cout << "[WARN]: Unrecognized key '" << key << "' found." << std::endl;
    }
}
