//
// Created by kazuto on 15/06/30.
//

#include <ros/ros.h>
#include <caffe_classifier/classify.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>

using namespace std;

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = true;

    ros::init(argc, argv, "test_publisher");

    ros::NodeHandle nh;
    ros::ServiceClient client = nh.serviceClient<caffe_classifier::classify>("caffe");

    cv::Mat img;
    cv::VideoCapture cap(0);
    CHECK(cap.isOpened()) << "Failed";

    while (ros::ok()) {
        cap >> img;
        CHECK(!img.empty()) << "Failed";

        cv::imshow("window", img);
        cv::waitKey(10);

        caffe_classifier::classify srv;
        srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();

        CHECK(client.call(srv)) << "Failed to call service";
        ROS_INFO_STREAM("response: " << srv.response.label);
    }

    return 0;
}