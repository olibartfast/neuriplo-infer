#include "ObjectDetectionApp.hpp"

int main(int argc, char *argv[]) {
    AppConfig config = parseCommandLineArguments(argc, argv);
    ObjectDetectionApp app(config);
    app.run();
    return 0;
}