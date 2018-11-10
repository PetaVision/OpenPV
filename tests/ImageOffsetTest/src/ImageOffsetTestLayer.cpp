#include "ImageOffsetTestLayer.hpp"

#include "ImageOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImageOffsetTestLayer::ImageOffsetTestLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

ImageOffsetTestLayer::~ImageOffsetTestLayer() {}

int ImageOffsetTestLayer::initialize(char const *name, HyPerCol *hc) {
   return ImageLayer::initialize(name, hc);
}

ActivityComponent *ImageOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageOffsetTestActivityBuffer>(getName(), parent);
}

void ImageOffsetTestLayer::setNontriggerDeltaUpdateTime(double dt) { mDeltaUpdateTime = 1.0; }

} /* namespace PV */
