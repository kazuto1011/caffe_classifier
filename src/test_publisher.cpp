//
// Created by kazuto on 15/06/30.
//

#include <ros/ros.h>
#include <caffe_classifier/classify.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>

using namespace std;

#define CAPTURE 0

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = true;

    ros::init(argc, argv, "test_publisher");

    ros::NodeHandle nh;
    ros::ServiceClient client = nh.serviceClient<caffe_classifier::classify>("caffe");

    cv::Mat img;

#if CAPTURE
    cv::VideoCapture cap(0);
    CHECK(cap.isOpened()) << "Failed";
#else
    img = cv::imread("/home/kazuto/cat.jpg");
    CHECK(!img.empty()) << "Failed";
#endif

    while (ros::ok()) {
#if CAPTURE
        cap >> img;
        CHECK(!img.empty()) << "Failed";
#endif
        caffe_classifier::classify srv;
        srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();

        CHECK(client.call(srv)) << "Failed to call service";
        ROS_INFO_STREAM("response: " << srv.response.label);

        cv::putText(img, srv.response.label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 200, 200),
                    1, CV_AA);
        cv::imshow("window", img);
        if (cv::waitKey(10) > 0) break;
    }

    return 0;
}