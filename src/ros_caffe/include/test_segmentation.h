#define USE_OPENCV 1
//#include <caffe/caffe.hpp>
#include <../caffe-segnet/include/caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono> //Just for time measurement

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;



class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);


  void Predict(const cv::Mat& img, string LUT_file);

 private:
  void SetMean(const string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void Visualization(Blob<float>* output_layer, string LUT_file);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;

};
