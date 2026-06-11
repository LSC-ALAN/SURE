#include "sure.h"

namespace realsee
{
    SURE::SURE(std::string &model_path)
    {
        // Create environment and session
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "sure");
        Ort::SessionOptions sessionOption;
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(IntraOpNumThreads);
        sessionOption.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);

        // // CUDA
        // OrtCUDAProviderOptions cuda_options;
        // cuda_options.device_id = 0;
        // cuda_options.arena_extend_strategy = 0;
        // cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        // cuda_options.do_copy_in_default_stream = 1;
        // sessionOption.AppendExecutionProvider_CUDA(cuda_options);

        // Start session
        session = new Ort::Session(env, model_path.c_str(), sessionOption);
        options = Ort::RunOptions{nullptr};
    }

    SURE::~SURE()
    {
        delete session;
    }

    bool SURE::pre_process(cv::Mat img0, cv::Mat img1, float *oneInput_)
    {
        // scale ratio
        ratioH = static_cast<float>(img0.rows) / static_cast<float>(inputH);
        ratioW = static_cast<float>(img0.cols) / static_cast<float>(inputW);

        // resize
        cv::resize(img0, img0, cv::Size(inputW, inputH));
        cv::resize(img1, img1, cv::Size(inputW, inputH));

        if (img0.channels() != 1)
        {
            cv::cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
        }

        // normalize
        img0.convertTo(img0, CV_32FC1, 1.0 / 255.0);
        img1.convertTo(img1, CV_32FC1, 1.0 / 255.0);

        // copy
        memcpy(oneInput_, img0.data, inputW * inputH * sizeof(float));
        memcpy(&oneInput_[inputW * inputH], img1.data, inputW * inputH * sizeof(float));

        return true;
    }

    bool SURE::match(cv::Mat &img0, cv::Mat &img1, std::vector<cv::KeyPoint> &kepts0, std::vector<cv::KeyPoint> &kepts1)
    {
        float *oneInput_ = new float[2 * inputW * inputH];

        pre_process(img0, img1, oneInput_);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            oneInput_,
            2 * inputW * inputH,
            inputNodeDims.data(),
            inputNodeDims.size());

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(inputTensor));

        auto outputTensor = session->Run(
            options,
            inputNodeNames.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            outputNodeNames.data(),
            outputNodeNames.size());

        Ort::Value &output_tensor = outputTensor[0];
        const float *output_data = output_tensor.GetTensorData<float>();

        post_process(img0, img1, output_data, kepts0, kepts1);

        return true;
    }

    // end to end
    void SURE::post_process(cv::Mat &img0, cv::Mat &img1, const float *output_data, std::vector<cv::KeyPoint> &kepts0, std::vector<cv::KeyPoint> &kepts1)
    {
        const int out_size = 11;
        for (int k = 0; k < topK; ++k)
        {
            float mkpts0_c_x = output_data[k * out_size + 0];
            float mkpts0_c_y = output_data[k * out_size + 1];
            float mkpts1_c_x = output_data[k * out_size + 2];
            float mkpts1_c_y = output_data[k * out_size + 3];
            float fine_offset01_x = output_data[k * out_size + 4];
            float fine_offset01_y = output_data[k * out_size + 5];
            float fine_offset10_x = output_data[k * out_size + 6];
            float fine_offset10_y = output_data[k * out_size + 7];
            float pred_score01 = output_data[k * out_size + 8];
            float pred_score10 = output_data[k * out_size + 9];
            float mconf = output_data[k * out_size + 10];

            if (mconf < conf_th || (pred_score01 < sigma_th && pred_score10 < sigma_th))
            {
                break;
            }

            // pred_score01 >= pred_score10
            float mkpts0_f_x = mkpts0_c_x;
            float mkpts0_f_y = mkpts0_c_y;
            float mkpts1_f_x = mkpts1_c_x + fine_offset01_x * local_resolution;
            float mkpts1_f_y = mkpts1_c_y + fine_offset01_y * local_resolution;

            if (pred_score01 < pred_score10)
            {
                mkpts0_f_x = mkpts0_c_x + fine_offset10_x * local_resolution;
                mkpts0_f_y = mkpts0_c_y + fine_offset10_y * local_resolution;
                mkpts1_f_x = mkpts1_c_x;
                mkpts1_f_y = mkpts1_c_y;
            }

            // filter border matches
            if (mkpts0_f_x < border_rm || mkpts0_f_x >= inputW - border_rm || mkpts0_f_y < border_rm || mkpts0_f_y >= inputH - border_rm || mkpts1_f_x < border_rm || mkpts1_f_x >= inputW - border_rm || mkpts1_f_y < border_rm || mkpts1_f_y >= inputH - border_rm)
            {
                continue;
            }
            // rescale output ketpoints coordinates to input image size
            auto cur_kp0 = cv::KeyPoint(cv::Point2f(mkpts0_f_x * ratioW, mkpts0_f_y * ratioH), 0);
            auto cur_kp1 = cv::KeyPoint(cv::Point2f(mkpts1_f_x * ratioW, mkpts1_f_y * ratioH), 0);

            kepts0.push_back(cur_kp0);
            kepts1.push_back(cur_kp1);
        }
    }

}