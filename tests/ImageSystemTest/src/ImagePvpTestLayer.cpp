#include "ImagePvpTestLayer.hpp"
#include "ImagePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpTestLayer::ImagePvpTestLayer(char const *name, HyPerCol *hc) { initialize(name, hc); }

ImagePvpTestLayer::~ImagePvpTestLayer() {}

int ImagePvpTestLayer::initialize(char const *name, HyPerCol *hc) {
   return InputLayer::initialize(name, hc);
}

ActivityComponent *ImagePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpTestBuffer>(getName(), parent);
}

} // end namespace PV
