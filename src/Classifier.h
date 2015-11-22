//
// Created by kazuto on 15/06/29.
//

#ifndef PINCH_CLASSIFIER_CLASSIFIER_H
#define PINCH_CLASSIFIER_CLASSIFIER_H

#include <ros/ros.h>
#include <ros/service_server.h>
#include <cv_bridge/cv_bridge.h>
#include <pinch_classifier/classify.h>
#include <caffe/caffe.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

using namespace std;
using boost::shared_ptr;

class Classifier {
private:
    ros::NodeHandle nh_;
    ros::ServiceServer server_;
    ros::AsyncSpinner spinner_;
    vector<string> synsets_;
    boost::shared_ptr<caffe::Net<float>> caffe_net_;
    boost::shared_ptr<caffe::MemoryDataLayer<float>> memory_data_layer_; //input layer
    boost::shared_ptr<caffe::Blob<float>> prob_; //output layer
public:
    Classifier(string proto_path, string model_path, string synsets_path);

    ~Classifier() { };

private:
    bool Classify(pinch_classifier::classify::Request &req,
                  pinch_classifier::classify::Response &res);

    bool InitSynsets(string synsets_path);

    bool LoadPretrainedModel(string proto_path, string model_path);

    cv_bridge::CvImagePtr ConvertMsg2Img(sensor_msgs::Image &image);

};

#endif //PINCH_CLASSIFIER_CLASSIFIER_H
