#include "HogSvmDetector.hpp"


std::string HogSvmDetector::modeName() const { return (m == Default ? "Default" : "Daimler"); }



std::vector<cv::Rect> HogSvmDetector::detect(cv::InputArray img)
  {
      // Run the detector with default parameters. to get a higher hit-rate
      // (and more false alarms, respectively), decrease the hitThreshold and
      // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
      std::vector<cv::Rect> found;
      if (m == Default)
          hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.70, 2, false);
      else if (m == Daimler)
          hog_d.detectMultiScale(img, found, 0.5, cv::Size(8,8), cv::Size(32,32), 1.70, 2, true);
      return found;
  }

void HogSvmDetector::adjustRect(cv::Rect & r) const
{
    // The HOG detector returns slightly larger rectangles than the real objects,
    // so we slightly shrink the rectangles to get a nicer output.
    r.x += cvRound(r.width*0.1);
    r.width = cvRound(r.width*0.8);
    r.y += cvRound(r.height*0.07);
    r.height = cvRound(r.height*0.8);
}

void HogSvmDetector::run_detection(cv::Mat& frame){
    int64 t = cv::getTickCount();
    std::vector<cv::Rect> found = detect(frame);
    t = cv::getTickCount() - t;

    // show the window
    {
        std::ostringstream buf;
        buf << "Mode: " << modeName() << " ||| "
            << "FPS: " << std::fixed << std::setprecision(1) << (cv::getTickFrequency() / (double)t);
        putText(frame, buf.str(), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    for (std::vector<cv::Rect>::iterator i = found.begin(); i != found.end(); ++i)
    {
        cv::Rect &r = *i;
        adjustRect(r);
        rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
    }
    if(found.size() > 0){
      time_t rawtime;
      struct tm * timeinfo;
      char buffer[80];
      time (&rawtime);
      timeinfo = localtime(&rawtime);
      strftime(buffer,sizeof(buffer),"%d%m%Y_%I%M%S",timeinfo);
      std::string fname(buffer);
      cv::imwrite(fname+".jpg", frame );
    }
}
