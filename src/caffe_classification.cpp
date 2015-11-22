//
// Created by kazuto on 15/07/10.
//

#include <ros/ros.h>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

const std::string proto_path = "/home/kazuto/catkin_ws/src/pinch_classifier/deploy.prototxt";
const std::string model_path = "/home/kazuto/catkin_ws/src/pinch_classifier/bvlc_reference_caffenet.caffemodel";
const std::string synset_file = "/home/kazuto/caffe/data/ilsvrc12/synset_words.txt";

bool initLabels(vector<string> &labels) {
    ifstream ifs(synset_file.c_str());
    if (!ifs) {
        cout << "Unable to open labels file " << synset_file;
        return false;
    }
    string line;
    while (std::getline(ifs, line))
        labels.push_back(string(line));
    return true;
}

int main(int argc, char **argv) {
#ifdef CPU_ONLY
    	std::cout<<"CPU_ONLY" << std::endl;
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif
    ros::init(argc, argv, "caffe_classification");

    vector<string> synsets;
    initLabels(synsets);

    //load pretrained model
    caffe::Net<float> *caffe_net;
    caffe_net = new caffe::Net<float>(proto_path, caffe::TEST);
    caffe_net->CopyTrainedLayersFrom(model_path);


    //input layer
    const boost::shared_ptr<caffe::MemoryDataLayer<float> > memory_data_layer =
            boost::static_pointer_cast<caffe::MemoryDataLayer<float> >(
                    caffe_net->layers()[0]/*caffe_net->layer_by_name("input")*/);
    if (!memory_data_layer) {
        cout << "the first layer is not caffe::MemoryDataLayer\n";
        return -1;
    }

    //output layer
    boost::shared_ptr<caffe::Blob<float>> prob = caffe_net->blob_by_name("prob");

    cv::VideoCapture cap("/home/kazuto/egocentric_video/read-a-book/2014112913204/1417234804024.avi");
    CHECK(cap.isOpened()) << "Failed";

    cv::Mat image;
    while (ros::ok()) {
        cap >> image;
        CHECK(!image.empty()) << "Failed";

        cv::resize(image, image, cv::Size(256, 256));
        std::vector<cv::Mat> images(1, image);
        std::vector<int> labels(1, 0);

        memory_data_layer->AddMatVector(images, labels);

        float loss;
        caffe_net->ForwardPrefilled(&loss);
        cout << "loss: " << loss << endl;

        float maxval = 0;
        int maxinx = 0;
        for (int i = 0; i < prob->count(); i++) {
            float val = prob->cpu_data()[i];
            if (val > maxval) {
                maxval = val;
                maxinx = i;
            }
        }

        std::cout << "Max value = " << maxval << ", Max index = " << synsets[maxinx] << endl;
    }

    return 0;
}
