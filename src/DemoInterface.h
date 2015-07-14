//
// Created by kazuto on 15/07/13.
//

#ifndef CAFFE_CLASSIFIER_DEMOINTERFACE_H
#define CAFFE_CLASSIFIER_DEMOINTERFACE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>


class DemoInterface {
private:
    ros::NodeHandle nh_;
    ros::ServiceClient client_;
    ros::AsyncSpinner spinner_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber it_subscriber_;
    int hsv[2][3];
public:

    DemoInterface();

    virtual ~DemoInterface();

    void ImgCallBack(const sensor_msgs::ImageConstPtr &msg);

private:
    cv::Mat ConvertMsg2Img(const sensor_msgs::ImageConstPtr &image);

    void ExtractColor(cv::Mat &hsv_img, cv::Mat &marker_img);
};


#endif //CAFFE_CLASSIFIER_DEMOINTERFACE_H
