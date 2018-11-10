#include "ImageLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/ImageActivityBuffer.hpp"

namespace PV {

ImageLayer::ImageLayer(char const *name, HyPerCol *hc) { initialize(name, hc); }

ImageLayer::~ImageLayer() {}

int ImageLayer::initialize(char const *name, HyPerCol *hc) {
   return InputLayer::initialize(name, hc);
}

ActivityComponent *ImageLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageActivityBuffer>(getName(), parent);
}

} // end namespace PV
