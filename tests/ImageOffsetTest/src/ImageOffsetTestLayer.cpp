#include "ImageOffsetTestLayer.hpp"

#include "ImageOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImageOffsetTestLayer::ImageOffsetTestLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ImageOffsetTestLayer::~ImageOffsetTestLayer() {}

void ImageOffsetTestLayer::initialize(char const *name, PVParams *params, Communicator *comm) {
   ImageLayer::initialize(name, params, comm);
}

ActivityComponent *ImageOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageOffsetTestActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

void ImageOffsetTestLayer::setNontriggerDeltaUpdateTime(double dt) { mDeltaUpdateTime = 1.0; }

} /* namespace PV */
