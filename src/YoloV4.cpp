#include "YoloV4.hpp"

    YoloV4::YoloV4(
 	    std::string modelConfiguration, 
        std::string modelBinary, 
        float confidenceThreshold,
        size_t network_width,
        size_t network_height    
    ) : 
		net_ {cv::dnn::readNetFromDarknet(modelConfiguration, modelBinary)}, 
        Detector{modelBinary, 
        confidenceThreshold,
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

std::vector<Detection> YoloV4::run_detection(const Mat& frame){
    cv::Mat inputBlob;
    cv::dnn::blobFromImage(frame, inputBlob, 1 / 255.F, cv::Size(network_width_, network_height_), cv::Scalar(), true, false, CV_32F);
    static std::vector<int> outLayers = net_.getUnconnectedOutLayers();
    static std::string outLayerType = net_.getLayer(outLayers[0])->type;
    std::vector<String> outNames = net_.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
	net_.setInput(inputBlob);
    net_.forward(outs, outNames);

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confidenceThreshold_)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    std::vector<Detection> detections;
    
    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region"))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confidenceThreshold_)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }

        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            const auto nmsThreshold = 0.4;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confidenceThreshold_, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                Detection d;
                size_t idx = nmsIndices[i];
                d.bbox = localBoxes[idx];
                d.score = localConfidences[idx];
                d.label = it->first;
                detections.emplace_back(d);

            }
        }
    }

    return detections;
}
