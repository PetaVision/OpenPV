#include "ImagePvpOffsetTestLayer.hpp"

#include "ImagePvpOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpOffsetTestLayer::ImagePvpOffsetTestLayer(const char *name, HyPerCol *hc) {
   initialize(name, hc);
}

ImagePvpOffsetTestLayer::~ImagePvpOffsetTestLayer() {}

int ImagePvpOffsetTestLayer::initialize(char const *name, HyPerCol *hc) {
   return PvpLayer::initialize(name, hc);
}

ActivityComponent *ImagePvpOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpOffsetTestActivityBuffer>(getName(), parent);
}

void ImagePvpOffsetTestLayer::setNontriggerDeltaUpdateTime(double dt) { mDeltaUpdateTime = 1.0; }

} /* namespace PV */
