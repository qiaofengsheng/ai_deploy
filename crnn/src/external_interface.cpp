#include "external_interface.hpp"
#include "frame_state_inference.hpp"

ExternalInterface::ExternalInterface()
{
}

ExternalInterface::~ExternalInterface()
{
}

ExternalInterface *ExternalInterface::createModel(const std::string &modelPath, string configPath,string dictPath)
{
    FrameStateInference *detector = new FrameStateInference();
    int status = detector->loadModel(modelPath, configPath,dictPath);
    if (status != 0)
    {
        delete detector;
        detector = nullptr;
    }
    return detector;
}
