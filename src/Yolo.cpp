#include "Yolo.hpp"

    Yolo::Yolo(
        const std::vector<std::string>& classNames,
 	    std::string modelConfiguration, 
        std::string modelBinary, 
        float confidenceThreshold,
        size_t network_width,
        size_t network_height    
    ) : 
		net_ {cv::dnn::readNetFromDarknet(modelConfiguration, modelBinary)}, 
        Detector{classNames, 
        modelConfiguration, modelBinary, confidenceThreshold,
        network_width,
        network_height}
	{
        if (net_.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "cfg-file:     " << modelConfiguration << std::endl;
            std::cerr << "weights-file: " << modelBinary << std::endl;
            exit(-1);
        }

	}

void Yolo::run_detection(const Mat& frame){
	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(network_width_, network_height_));
	cv::Mat inputBlob = blobFromImage(resized, 1 / 255.F); //Convert Mat to batch of images

	net_.setInput(inputBlob, "data");                //set the network input
	cv::Mat detectionMat = net_.forward("detection_out");	//compute output

	for (int i = 0; i < detectionMat.rows; i++)
	{
	    const int probability_index = 5;
	    const int probability_size = detectionMat.cols - probability_index;
	    float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

	    size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
	    float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

	    if (confidence > confidenceThreshold_)
	    {
	        float x = detectionMat.at<float>(i, 0);
	        float y = detectionMat.at<float>(i, 1);
	        float width = detectionMat.at<float>(i, 2);
	        float height = detectionMat.at<float>(i, 3);
	        float xLeftBottom = (x - width / 2) * frame.cols;
	        float yLeftBottom = (y - height / 2) * frame.rows;
	        float xRightTop = (x + width / 2) * frame.cols;
	        float yRightTop = (y + height / 2) * frame.rows;





	        std::cout << "Class: " << classNames_[objectClass] << std::endl;
	        std::cout << "Confidence: " << confidence << std::endl;

            std::ostringstream ss;
            ss << confidence;
            std::string conf(ss.str());

            cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

            cv::rectangle(frame, object, Scalar(0, 255, 0));
            std::string label = std::string(classNames_[objectClass]) + ": " + conf;
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height),
                                  cv::Size(labelSize.width, labelSize.height + baseLine)),
                      Scalar(255, 255, 255), cv::FILLED);
            cv::putText(frame, label, Point(xLeftBottom, yLeftBottom),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));  
	    }
	}

}