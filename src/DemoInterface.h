//
// Created by kazuto on 15/07/13.
//

#ifndef CAFFE_CLASSIFIER_DEMOINTERFACE_H
#define CAFFE_CLASSIFIER_DEMOINTERFACE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

class DemoInterface {
private:
    ros::NodeHandle nh_;
    ros::ServiceClient client_;
    ros::AsyncSpinner spinner_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber it_subscriber_;
    image_transport::Subscriber drawer_;
    int hsv[2][3];
    cv::Mat raw_img_;
    cv::Point point_rec_1_;
    cv::Point point_rec_2_;
    cv::Point point_;
    string result_;
    bool draw_;
    bool video_;
    int delay_;
    cv::VideoWriter writer_;
public:

    DemoInterface();

    virtual ~DemoInterface();

    void ImgCallBack(const sensor_msgs::ImageConstPtr &msg);

    void DrawCallBack(const sensor_msgs::ImageConstPtr &msg);

private:
    cv::Mat ConvertMsg2Img(const sensor_msgs::ImageConstPtr &image);

    void ExtractColor(cv::Mat &hsv_img, cv::Mat &marker_img);
};

#endif //CAFFE_CLASSIFIER_DEMOINTERFACE_H
