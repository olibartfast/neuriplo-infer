#include "MobileNetSSD.hpp"



MobileNetSSD::MobileNetSSD(
        const std::vector<std::string>& classNames,
        std::string modelConfiguration, 
        std::string modelBinary,      
        float confidenceThreshold,    
        size_t network_width,
        size_t network_height,
        float inScaleFactor,
        float meanVal) : 
        inScaleFactor_ {inScaleFactor},
        meanVal_ {meanVal},
        net_ {cv::dnn::readNetFromCaffe(modelConfiguration, modelBinary)}, 
        Detector{classNames, 
        modelConfiguration, modelBinary, confidenceThreshold,
        network_width,
        network_height}
{       

}




void MobileNetSSD::run_detection(cv::Mat& frame)
{
  // Create a 4D blob from a frame.

    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor_,
                                  cv::Size(network_width_, network_height_), meanVal_, false); //Convert Mat to batch of images
    net_.setInput(inputBlob, "data"); //set the network input
    cv::Mat detection = net_.forward("detection_out"); //compute output
    std::vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = net_.getPerfProfile(layersTimings) / freq;
    std::cout << "Inference time, ms: " << time << std::endl;

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold_)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            std::ostringstream ss;
            ss << confidence;
            std::string conf(ss.str());

            cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

            cv::rectangle(frame, object, cv::Scalar(0, 255, 0));
            std::string label = std::string(classNames_[objectClass]) + ": " + conf;
            int baseLine = 0;
            cv::Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                  Size(labelSize.width, labelSize.height + baseLine)),
                      Scalar(255, 255, 255), cv::FILLED);
            cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));     
        }
    }

}
