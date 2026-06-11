#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include "sure/sure.h"

using namespace std;
using namespace cv;

typedef unsigned char uchar;

cv::Mat drawKpMatches(const std::vector<cv::KeyPoint> &kepts0,
                      const std::vector<cv::KeyPoint> &kepts1,
                      const cv::Mat &rgb1, const cv::Mat &rgb2)
{
    cv::Mat res;

    int rWidth = rgb1.cols;
    int rHeight = rgb1.rows;

    int circle_r = rWidth / 640;

    float rW = static_cast<float>(rWidth) / static_cast<float>(rgb1.cols);
    float rH = static_cast<float>(rHeight) / static_cast<float>(rgb1.rows);

    res = cv::Mat(cv::Size(2 * rWidth, rHeight), CV_8UC3);

    cv::Mat rgb1_cam, rgb2_cam;
    cv::resize(rgb1, rgb1_cam, cv::Size(rWidth, rHeight));
    cv::resize(rgb2, rgb2_cam, cv::Size(rWidth, rHeight));

    if (rgb1.channels() != 3)
    {
        cv::cvtColor(rgb1_cam, rgb1_cam, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rgb2_cam, rgb2_cam, cv::COLOR_GRAY2BGR);
    }

    cv::resize(rgb1_cam, rgb1_cam, cv::Size(rWidth, rHeight));
    cv::resize(rgb2_cam, rgb2_cam, cv::Size(rWidth, rHeight));
    rgb1_cam.copyTo(res(cv::Rect(0, 0, rWidth, rHeight)));
    rgb2_cam.copyTo(res(cv::Rect(rWidth, 0, rWidth, rHeight)));

    cv::RNG &rng = cv::theRNG();
    cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));
    for (size_t i = 0; i < kepts0.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));
        cv::Point2f pt = kepts0[i].pt;

        pt.x *= rW;
        pt.y *= rH;

        cv::circle(res, pt, circle_r, color);
        cv::Point2f pt2 = kepts1[i].pt;

        pt2.x *= rW;
        pt2.y *= rH;
        pt2.x += rWidth;
        cv::circle(res, pt2, circle_r, color);
    }

    for (size_t i = 0; i < kepts0.size(); i++)
    {
        cv::RNG &rng = cv::theRNG();
        cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));
        cv::line(res, cv::Point(kepts0[i].pt.x * rW, kepts0[i].pt.y * rH),
                 cv::Point(kepts1[i].pt.x * rW + rWidth, kepts1[i].pt.y * rH), color);
    }

    return res;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        cout << "usage: ./demo img0_path img1_path out_path model_path\n";
    }

    std::string img_path0 = argv[1];
    std::string img_path1 = argv[2];
    std::string output_path = argv[3];
    std::string model_path = argv[4];

    // init model
    auto matcher = std::make_shared<realsee::SURE>(model_path);

    // read images
    cv::Mat rgb0 = cv::imread(img_path0);
    cv::Mat rgb1 = cv::imread(img_path1);

    std::vector<cv::KeyPoint> kepts0;
    std::vector<cv::KeyPoint> kepts1;

    // inference
    matcher->match(rgb0, rgb1, kepts0, kepts1);
    std::cout << "match num: " << kepts0.size() << std::endl;


    // draw matches
    cv::Mat result = drawKpMatches(kepts0, kepts1, rgb0, rgb1);
    cv::imwrite(output_path, result);

    return 0;
}
