#include "ImagePvpOffsetTestLayer.hpp"

#include "ImagePvpOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpOffsetTestLayer::ImagePvpOffsetTestLayer(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

ImagePvpOffsetTestLayer::~ImagePvpOffsetTestLayer() {}

void ImagePvpOffsetTestLayer::initialize(char const *name, PVParams *params, Communicator *comm) {
   PvpLayer::initialize(name, params, comm);
}

ActivityComponent *ImagePvpOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpOffsetTestActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

void ImagePvpOffsetTestLayer::setNontriggerDeltaUpdateTime(double dt) { mDeltaUpdateTime = 1.0; }

} /* namespace PV */
