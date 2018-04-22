#include "HogSvmDetector.h"


string HogSvmDetector::modeName() const { return (m == Default ? "Default" : "Daimler"); }



vector<Rect> HogSvmDetector::detect(InputArray img)
  {
      // Run the detector with default parameters. to get a higher hit-rate
      // (and more false alarms, respectively), decrease the hitThreshold and
      // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
      vector<Rect> found;
      if (m == Default)
          hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.70, 2, false);
      else if (m == Daimler)
          hog_d.detectMultiScale(img, found, 0.5, Size(8,8), Size(32,32), 1.70, 2, true);
      return found;
  }

void HogSvmDetector::adjustRect(Rect & r) const
{
    // The HOG detector returns slightly larger rectangles than the real objects,
    // so we slightly shrink the rectangles to get a nicer output.
    r.x += cvRound(r.width*0.1);
    r.width = cvRound(r.width*0.8);
    r.y += cvRound(r.height*0.07);
    r.height = cvRound(r.height*0.8);
}

Mat HogSvmDetector::run_detection(Mat frame){
    int64 t = getTickCount();
    vector<Rect> found = detect(frame);
    t = getTickCount() - t;

    // show the window
    {
        ostringstream buf;
        buf << "Mode: " << modeName() << " ||| "
            << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
        putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
    }
    for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
    {
        Rect &r = *i;
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
      string fname(buffer);
      imwrite(fname+".jpg", frame );
    }
    return frame;
}
