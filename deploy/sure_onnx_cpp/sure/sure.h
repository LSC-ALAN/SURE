/*
 * filename: sure.h
 * author: Xi Li
 * email: xi-li@foxmail.com
 * create time: 2025-03-19
 * update time: 2025-03-19
 * description: C++ inference demo for SURE.
 */
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


namespace realsee
{
    class SURE
    {
    public:
        SURE(std::string &model_path);

        ~SURE();

        bool pre_process(cv::Mat img0, cv::Mat img1, float *oneInput_);
        bool match(cv::Mat &img0, cv::Mat &img1, std::vector<cv::KeyPoint> &kepts0, std::vector<cv::KeyPoint> &kepts1);
        void post_process(cv::Mat &img0, cv::Mat &img1, const float *output_data, std::vector<cv::KeyPoint> &kepts0, std::vector<cv::KeyPoint> &kepts1);

    private:
        Ort::Env env;
        Ort::Session *session;
        Ort::RunOptions options;

        std::vector<const char *> inputNodeNames = {"input"};
        std::vector<const char *> outputNodeNames = {"output"};

        // scale ratio
        float ratioH = 1.0f;
        float ratioW = 1.0f;

        // inference size
        const int inputW = 640;
        const int inputH = 480;

        // concat two gray images on the second dimension
        std::vector<int64_t> inputNodeDims = {1, 2, inputH, inputW};

        // onnxruntime configs
        const int IntraOpNumThreads = 8;

        // sure configs
        const int topK = 1680;                      // depend on exported onnx model
        const int local_resolution = 8;             // default
        const int border_rm = 2 * local_resolution; // Remove matches at image boundaries
        const float conf_th = 0.2;                  // coarse threshold
        const float sigma_th = 1e-6;                // fine threshold
    };
}
