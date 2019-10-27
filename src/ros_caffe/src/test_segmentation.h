#define USE_OPENCV 1
#include <caffe/caffe.hpp>

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
//#include <chrono> //Just for time measurement

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;



class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);
  std::vector<Prediction> Classify(const std::vector<float> output, int N);
  void Predict(const cv::Mat& img, string LUT_file, cv::Mat &output_img, cv::Mat &output);

  cv::Size input_geometry_;

  // scale parameters from resize image to transfer img kordinates
  double scale_x;
  double scale_y;

 private:
  void SetMean(const string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  cv::Mat Visualization(cv::Mat prediction_map, string LUT_file);

 private:
  shared_ptr<Net<float> > net_;
  //cv::Size input_geometry_;
  int num_channels_;

};

//class to put image through fcn-network
Classifier::Classifier(const string& model_file,
                       const string& trained_file) {


  Caffe::set_mode(Caffe::CPU);

  /* Load the network. */
  std::cout << model_file << std::endl;
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}


void Classifier::Predict(const cv::Mat& img, string LUT_file, cv::Mat& output_img, cv::Mat& output) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);


  //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

  net_->Forward();

  //std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  //std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << " sec" <<std::endl; //Just for time measurement


  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
//  const float* begin = output_layer->cpu_data();
//  const float* end = begin + output_layer->channels();
  /* Copy the output layer to a std::vector */
  int width = output_layer->width();
  int height = output_layer->height();
  int channels = output_layer->channels();
  int num = output_layer->num();

  std::cout << "output_blob(n,c,h,w) = " << num << ", " << channels << ", "
              << height << ", " << width << std::endl;

  // compute argmax
  //cv::Mat *matrix = new cv::Mat(3, sizes, CV_32FC1, cv::Scalar(0));
  //cv::Mat all_channel (3, (height, width, channels), CV_32FC1, const_cast<float *>(output_layer->cpu_data()));
  cv::Mat class_each_row (channels, width*height, CV_32FC1, const_cast<float *>(output_layer->cpu_data()));
  class_each_row = class_each_row.t(); // transpose to make each row with all probabilities
  cv::Point maxId;    // point [x,y] values for index of max
  double maxValue;    // the holy max value itself
  cv::Mat prediction_map(height, width, CV_8UC1);
  for (int i=0;i<class_each_row.rows;i++){
      minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);
      prediction_map.at<uchar>(i) = maxId.x;
  }

  //cv::Mat class_img = Visualization(prediction_map,LUT_file);
  output_img = prediction_map;
  output = class_each_row;
}


cv::Mat Classifier::Visualization(cv::Mat prediction_map, string LUT_file) {

  cv::cvtColor(prediction_map.clone(), prediction_map, CV_GRAY2BGR);
  cv::Mat label_colours = cv::imread(LUT_file,1);
  cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);
  cv::Mat output_image;
  LUT(prediction_map, label_colours, output_image);

//  cv::imshow( "Display window", output_image);
//  cv::waitKey(0);

  return output_image;

}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  this->scale_x = sample_resized.cols/sample.cols;
  this->scale_y = sample_resized.rows/sample.rows;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}


//class is done
