//
// Created by kazuto on 15/07/13.
//

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <glog/logging.h>
#include <pinch_classifier/classify.h>
#include "DemoInterface.h"
#include "Labeling.h"

using namespace std;
#define DELAY 5
#define SUM_MIN 300

DemoInterface::DemoInterface() : it_(nh_), spinner_(0), draw_(false), delay_(DELAY), video_(true) {
    ROS_INFO_STREAM("DemoInterface Constructor");
    FLAGS_logtostderr = true;

    image_transport::TransportHints hints("compressed", ros::TransportHints());
    it_subscriber_ = it_.subscribe("/camera/rgb/image_raw", 1, &DemoInterface::ImgCallBack, this, hints);
    drawer_ = it_.subscribe("/camera/rgb/image_raw", 1, &DemoInterface::DrawCallBack, this, hints);
    client_ = nh_.serviceClient<pinch_classifier::classify>("caffe");

    cv::namedWindow("Raw");
    cv::namedWindow("Output");
    cv::namedWindow("Marker");
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
    if (raw_img_.empty())
        return;

    cv::Rect roi(0, 0, raw_img_.cols, raw_img_.rows);
    cv::Mat marker_img = cv::Mat::zeros(raw_img_.size(), CV_8UC1);

    //一度平滑化した後，HSV色空間に変換
    cv::Mat hsv_img;
    cv::medianBlur(raw_img_, hsv_img, 13);
    cv::cvtColor(hsv_img, hsv_img, CV_BGR2HSV);

    //黄色の領域を抽出
    ExtractColor(hsv_img, marker_img);

    //ラベリング
    cv::Mat label(raw_img_.size(), CV_16SC1);
    LabelingBS labeling;
    labeling.Exec(marker_img.data, (short *) label.data, raw_img_.cols, raw_img_.rows, false, 0);

    cv::Mat out_img(raw_img_.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    int num_regions = labeling.GetNumOfRegions();

    //ラベリング数が2以上，10未満の時
    if (num_regions > 1 && num_regions < 10) {
        cv::Point point[2];
        cv::Point max_pt[2] = {};
        int max_idx[2] = {0, 1};
        int max_sum[2] = {};
        int num_pt = 0;

        //面積最大の要素を2つ計算する
        for (int i = 0; i < num_regions; i++) {
            int sum = 0;
            cv::Point pt(0, 0);
            cv::Mat_<uchar> labelarea;
            cv::compare(label, i + 1, labelarea, CV_CMP_EQ);

            for (int y = 0; y < labelarea.rows; y++) {
                const uchar *labelarea_row = labelarea.ptr<uchar>(y);
                for (int x = 0; x < labelarea.cols; x++) {
                    if (labelarea_row[x] > 250) {
                        sum++;
                        pt.x += x;
                        pt.y += y;
                    }
                }
            }

            for (int j = 0; j < 2; j++) {
                if (sum > max_sum[j]) {
                    if (pt.x == 0 && pt.y == 0)
                        break;
                    else if (sum < SUM_MIN)
                        break;
                    for (int n = 1; n > j; n--) {
                        max_idx[n] = max_idx[n - 1];
                        max_sum[n] = max_sum[n - 1];
                        max_pt[n] = max_pt[n - 1];
                    }
                    max_idx[j] = i;
                    max_sum[j] = sum;
                    max_pt[j] = pt;
                    num_pt++;
                    break;
                }
            }
        }

        if (num_pt != 2)
            return;

        for (int i = 0; i < 2; i++) {
            int sum = max_sum[i];
            int idx = max_idx[i];
            point[i] = max_pt[i];

            cv::Mat_<uchar> labelarea;
            cv::compare(label, idx + 1, labelarea, CV_CMP_EQ);
            cv::Mat color(raw_img_.size(), CV_8UC3, cv::Scalar(255, 255, 255));

            if (sum > 0) {
                point[i].x /= sum;
                point[i].y /= sum;
                color.copyTo(out_img, labelarea);
                cv::circle(out_img, point[i], 20, cv::Scalar(0, 200, 0), 8, 8);
            }
        }

        //指定領域の行列のサイズと中点
        int row_diff = abs(point[0].y - point[1].y);
        int col_diff = abs(point[0].x - point[1].x);
        int row_center = (point[0].y + point[1].y) / 2;
        int col_center = (point[0].x + point[1].x) / 2;
        int length = 0;

        //縦幅と横幅の大きい方を正方形の一辺に取る
        if (row_diff > col_diff)
            length = row_diff;
        else
            length = col_diff;

        roi.x = col_center - length / 2;
        roi.y = row_center - length / 2;
        roi.width = length;
        roi.height = length;

        //境界付近の処理
        if (roi.x + roi.width > raw_img_.cols)
            roi.width = raw_img_.cols - roi.x;
        if (roi.y + roi.height > raw_img_.rows)
            roi.height = raw_img_.rows - roi.y;
        if (roi.x < 0) {
            roi.width += roi.x;
            roi.x = 0;
        }
        if (roi.y < 0) {
            roi.height += roi.y;
            roi.y = 0;
        }

        if (roi.x || roi.y || roi.width || roi.height) {
            point_rec_1_ = cv::Point(col_center - length / 2, row_center - length / 2);
            point_rec_2_ = cv::Point(col_center + length / 2, row_center + length / 2);

            //受信画像の指定領域をROSメッセージにセット
            pinch_classifier::classify srv;
            srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", raw_img_(roi)).toImageMsg();

            //Caffeサーバを呼び出す
            CHECK(client_.call(srv)) << "Failed to call service";

            //返ってきた結果を整形
            result_ = ConvertResponse(srv.response.label);
            point_ = cv::Point(roi.x, roi.y - 20);
            draw_ = true;
            delay_ = DELAY;
        }
    }
    else {
        if (--delay_ == 0) {
            draw_ = false;
            delay_ = DELAY;
        }
    }

    cv::rectangle(out_img, point_rec_1_, point_rec_2_, cv::Scalar(0, 200, 0), 3, 4);
    cv::imshow("Yellow", out_img);
    cv::imshow("Marker", marker_img);
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


string DemoInterface::ConvertResponse(string srv_label) {
    vector<string> elems;
    string item, response = srv_label;
    char delim = ',';
    for (char ch: response) {
        if (ch == delim) {
            if (!item.empty())
                elems.push_back(item);
            item.clear();
        }
        else
            item += ch;
    }

    if (!item.empty())
        elems.push_back(item);

    response = elems[0];
    elems.clear();
    delim = ' ';
    for (char ch: response) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
                delim = '\0';
            }
            item.clear();
        }
        else
            item += ch;
    }

    if (!item.empty())
        elems.push_back(item);

    return elems[1];
}

void DemoInterface::DrawCallBack(const sensor_msgs::ImageConstPtr &msg) {
    //受信画像をcv::Matに変換
    raw_img_ = this->ConvertMsg2Img(msg);
    cv::Mat test = this->ConvertMsg2Img(msg);
    //最初の一回のみ
    if (video_) {
        writer_.open("/home/kazuto/Desktop/videofile.avi", CV_FOURCC_DEFAULT, 30.0, raw_img_.size());
        if (!writer_.isOpened()) {
            LOG(INFO) << "not opend";
            throw;
        }
        video_ = false;
    }

    //画像に対する識別結果と指定窓の重畳表示
    if (draw_) {
        cv::putText(raw_img_, result_, point_, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 200, 0),
                    2, CV_AA);
        cv::rectangle(raw_img_, point_rec_1_, point_rec_2_, cv::Scalar(0, 200, 0), 3, 4);
    }

    if (!raw_img_.empty()) {
        cv::imshow("Output", raw_img_);
        writer_ << raw_img_;
        cv::imshow("Raw", test);
    }
}

