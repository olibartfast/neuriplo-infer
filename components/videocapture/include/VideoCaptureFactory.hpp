#pragma once
#include "VideoCaptureInterface.hpp"
#ifdef USE_GSTREAMER
#include "GStreamerCapture.hpp"
#else
#include "OpenCVCapture.hpp"
#endif

 std::unique_ptr<VideoCaptureInterface> createVideoInterface() 
 {
        #ifdef USE_GSTREAMER
            return std::make_unique<GStreamerCapture>();
        #else
            return std::make_unique<OpenCVCapture>();
        #endif
}