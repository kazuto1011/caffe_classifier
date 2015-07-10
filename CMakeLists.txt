cmake_minimum_required(VERSION 2.8.3)
project(caffe_classifier)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#add_definitions(-DCPU_ONLY=1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_generation
)
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET)

set(Caffe_INCLUDE_DIRS ~/caffe/distribute/include)
set(Caffe_LIBS ~/caffe/build/lib)

add_service_files(
  FILES
  classify.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
    INCLUDE_DIRS include
#  LIBRARIES caffe_classifier
    CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge message_runtime
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Caffe_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

###############################
## sample cpp classification ##
###############################
link_directories(${Caffe_LIBS})
add_executable(caffe_classification src/caffe_classification.cpp)
add_dependencies(caffe_classification caffe_classifier_generate_messages_cpp)
target_link_libraries(caffe_classification
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBS}
  caffe
  glog
  gflags
)

###############################
##   caffe classifier node   ##
###############################
link_directories(${Caffe_LIBS})
add_executable(caffe_classification_server
  src/caffe_classification_server.cpp
  src/Classifier.cpp
)
add_dependencies(caffe_classification_server caffe_classifier_generate_messages_cpp)
target_link_libraries(caffe_classification_server
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBS}
  caffe
  glog
  gflags
)

###############################
##   test image publisher    ##
###############################
add_executable(test_client src/test_client.cpp)
add_dependencies(test_client caffe_classifier_generate_messages_cpp)
target_link_libraries(test_client
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  glog
  gflags
)