#include "Yolo.h"

void Yolo::init(const char** coco_classes, string modelConfiguration, string modelBinary, 
	const size_t network_width,
    const size_t network_height,
	float confidenceThreshold){

	coco_classes_ = coco_classes;
    network_width_ = network_width;
    network_height_ = network_height;
    confidenceThreshold_ = confidenceThreshold;

    //! [Initialize network]
    net_ = readNetFromDarknet(modelConfiguration, modelBinary);
    //! [Initialize network]
    if (net_.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }

}

void Yolo::run_yolo(Mat& frame){
	//! [Resizing without keeping aspect ratio]
	Mat resized;
	resize(frame, resized, cv::Size(network_width_, network_height_));
	//! [Prepare blob]
	Mat inputBlob = blobFromImage(resized, 1 / 255.F); //Convert Mat to batch of images
	//! [Prepare blob]

	//! [Set input blob]
	net_.setInput(inputBlob, "data");                //set the network input
	//! [Set input blob]

	//! [Make forward pass]
	cv::Mat detectionMat = net_.forward("detection_out");	//compute output
	//! [Make forward pass]
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



	        String label = String(coco_classes_[objectClass]) + ": " + confidence;

	        std::cout << "Class: " << coco_classes_[objectClass] << std::endl;
	        std::cout << "Confidence: " << confidence << std::endl;

	        std::cout << " " << xLeftBottom
	            << " " << yLeftBottom
	            << " " << xRightTop
	            << " " << yRightTop << std::endl;

	        /*Rect object((int)xLeftBottom, (int)yLeftBottom,
	            (int)(xRightTop - xLeftBottom),
	            (int)(yRightTop - yLeftBottom));

	        rectangle(frame, object, Scalar(0, 255, 0));*/
	        int baseLine = 0;
	        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	        rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
	                              Size(labelSize.width, labelSize.height + baseLine)),
	                  Scalar(255, 255, 255), CV_FILLED);
	        putText(frame, label, Point(xLeftBottom, yLeftBottom),
	                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));    
	    }
	}

}