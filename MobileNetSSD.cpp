#include "MobileNetSSD.h"



void MobileNetSSD::init(const char** classNames_,
        size_t inWidth_, 
        size_t inHeight_, 
        float inScaleFactor_,
        float meanVal_, 
        int frameWidth_, 
        int frameHeight_,  
        float confidenceThreshold_,       
        String modelConfiguration_, 
        String modelBinary_){

  classNames = classNames_;
  inWidth = inWidth_;
  inHeight = inHeight_;
  inScaleFactor = inScaleFactor_;
  meanVal = meanVal_;
  modelConfiguration = modelConfiguration_;
  modelBinary = modelBinary_;
  WHRatio = inWidth / (float)inHeight;

  frameWidth = frameWidth_;
  frameHeight = frameHeight_;

  confidenceThreshold = confidenceThreshold_;

  inVideoSize = Size(frameWidth, frameHeight);

  if (inVideoSize.width / (float)inVideoSize.height > WHRatio)
  {
    cropSize = Size(static_cast<int>(inVideoSize.height * WHRatio),
                    inVideoSize.height);
  }
  else
  {
    cropSize = Size(inVideoSize.width,
                        static_cast<int>(inVideoSize.width / WHRatio));
  }
  net = readNetFromCaffe(modelConfiguration, modelBinary);

  crop = Rect(Point((inVideoSize.width - cropSize.width) / 2,
                    (inVideoSize.height - cropSize.height) / 2),
              cropSize);

}




Mat MobileNetSSD::run_ssd(Mat frame){
  // Create a 4D blob from a frame.

    Mat inputBlob = blobFromImage(frame, inScaleFactor,
                                  Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images
    //! [Prepare blob]

    //! [Set input blob]
    net.setInput(inputBlob, "data"); //set the network input
    //! [Set input blob]

    //! [Make forward pass]
    Mat detection = net.forward("detection_out"); //compute output
    //! [Make forward pass]

    std::vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;
    cout << "Inference time, ms: " << time << endl;

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

   // frame = frame(crop);

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            ostringstream ss;
            ss << confidence;
            String conf(ss.str());

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

            rectangle(frame, object, Scalar(0, 255, 0));
            String label = String(classNames[objectClass]) + ": " + conf;
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                  Size(labelSize.width, labelSize.height + baseLine)),
                      Scalar(255, 255, 255), CV_FILLED);
            putText(frame, label, Point(xLeftBottom, yLeftBottom),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
        }
    }

  return frame;

}
