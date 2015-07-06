//
// Created by kazuto on 15/06/30.
//

#include "Classifier.h"

const std::string proto_path = "/home/kazuto/catkin_ws/src/caffe_classifier/deploy.prototxt";
const std::string model_path = "/home/kazuto/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
const std::string synset_file = "/home/kazuto/caffe/data/ilsvrc12/synset_words.txt";

int main(int argc, char **argv) {
#ifdef CPU_ONLY
    ROS_INFO_STREAM("CPU_ONLY");
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr = false;

    ros::init(argc, argv, "caffe_classifier");

    boost::shared_ptr<Classifier> classifier;
    classifier.reset(new Classifier(proto_path, model_path, synset_file));

    ros::waitForShutdown();
    return 0;
}