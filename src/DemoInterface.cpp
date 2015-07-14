//
// Created by kazuto on 15/07/13.
//

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <glog/logging.h>
#include <caffe_classifier/classify.h>
#include "DemoInterface.h"

DemoInterface::DemoInterface() : it_(nh_), spinner_(0) {
    ROS_INFO_STREAM("DemoInterface Constructor");

    image_transport::TransportHints hints("compressed", ros::TransportHints());
    it_subscriber_ = it_.subscribe("/camera/rgb/image_raw", 1, &DemoInterface::ImgCallBack, this, hints);
    client_ = nh_.serviceClient<caffe_classifier::classify>("caffe");

    cv::namedWindow("Xtion");
    cv::namedWindow("Yellow");
    cv::startWindowThread();

    spinner_.start();

    hsv[0][0] = 15;
    hsv[1][0] = 35;
    hsv[0][1] = 150;
    hsv[1][1] = 255;
    hsv[0][2] = 150;
    hsv[1][2] = 255;
}

DemoInterface::~DemoInterface() {
    ROS_INFO_STREAM("DemoInterface Destructor");
    it_subscriber_.shutdown();
    client_.shutdown();
}

void DemoInterface::ImgCallBack(const sensor_msgs::ImageConstPtr &msg) {
    cv::Mat raw_img = this->ConvertMsg2Img(msg);
    cv::Mat marker_img = cv::Mat::zeros(raw_img.size(), CV_8UC1);

    cv::Mat hsv_img;
    cv::medianBlur(raw_img, hsv_img, 13);
    cv::cvtColor(hsv_img, hsv_img, CV_BGR2HSV);

    ExtractColor(hsv_img, marker_img);

    caffe_classifier::classify srv;
    srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", raw_img).toImageMsg();

    CHECK(client_.call(srv)) << "Failed to call service";
    ROS_INFO_STREAM("response: " << srv.response.label);

    cv::putText(raw_img, srv.response.label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 200, 200),
                1, CV_AA);
    cv::imshow("Xtion", raw_img);
    cv::imshow("Yellow", marker_img);
}

cv::Mat DemoInterface::ConvertMsg2Img(const sensor_msgs::ImageConstPtr &image) {
    cv::Mat img;
    try {
        img = cv_bridge::toCvCopy(image, "bgr8")->image;
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    return img;
}

void DemoInterface::ExtractColor(cv::Mat &hsv_img, cv::Mat &marker_img) {
    for (int y = 0; y < hsv_img.rows; y++) {
        const cv::Vec3b *hsv_row = hsv_img.ptr<cv::Vec3b>(y);
        uchar *marker_img_row = marker_img.ptr<uchar>(y);
        for (int x = 0; x < hsv_img.cols; x++) {
            if (
                    hsv_row[x][0] >= hsv[0][0] &&
                    hsv_row[x][0] <= hsv[1][0] &&
                    hsv_row[x][1] >= hsv[0][1] &&
                    hsv_row[x][1] <= hsv[1][1] &&
                    hsv_row[x][2] >= hsv[0][2] &&
                    hsv_row[x][2] <= hsv[1][2]) {
                marker_img_row[x] = 255;
            }
        }
    }
}
