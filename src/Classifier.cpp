//
// Created by kazuto on 15/06/29.
//

#include <fstream>
#include <caffe/net.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "Classifier.h"

using namespace std;
using boost::shared_ptr;
using boost::static_pointer_cast;

Classifier::Classifier(string proto_path, string model_path, string synsets_path) : spinner_(0) {
    CHECK(InitSynsets(synsets_path)) << "Unable to open labels file ";
    CHECK(LoadPretrainedModel(proto_path, model_path)) << "Unable to load model";

    memory_data_layer_ = boost::static_pointer_cast<caffe::MemoryDataLayer<float >>(caffe_net_->layers()[0]);
    server_ = nh_.advertiseService("caffe", &Classifier::Classify, this);
    spinner_.start();
};

bool Classifier::Classify(caffe_classifier::classify::Request &req,
                          caffe_classifier::classify::Response &res) {

    cv::Mat image = ConvertMsg2Img(req.image)->image;
    cv::resize(image, image, cv::Size(256, 256));

    std::vector<cv::Mat> images(1, image);
    std::vector<int> labels(1, 0);

    memory_data_layer_->AddMatVector(images, labels);

    float loss;
    caffe_net_->ForwardPrefilled(&loss);
    ROS_INFO_STREAM("loss: " << loss);

    prob_.reset(caffe_net_->output_blobs()[0]);

    float max_value = 0;
    int max_index = 0;
    for (int i = 0; i < prob_->count(); i++) {
        float value = prob_->cpu_data()[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }

    ROS_INFO_STREAM("Max value = " << max_value << ", Max index = " << synsets_[max_index]);
    res.label = synsets_[max_index];
    return true;
};

bool Classifier::InitSynsets(string synsets_path) {
    ifstream ifs(synsets_path.c_str());
    if (!ifs)
        return false;
    string line;
    while (std::getline(ifs, line))
        synsets_.push_back(string(line));
    return true;
};

bool Classifier::LoadPretrainedModel(string proto_path, string model_path) {
    caffe_net_.reset(new caffe::Net<float>(proto_path, caffe::TEST));
    if (!caffe_net_)
        return false;
    caffe_net_->CopyTrainedLayersFrom(model_path);
    return true;
};

cv_bridge::CvImagePtr Classifier::ConvertMsg2Img(sensor_msgs::Image &image) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image, image.encoding);
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return NULL;
    }
    return cv_ptr;
}