#include "ImageLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/ImageActivityBuffer.hpp"

namespace PV {

ImageLayer::ImageLayer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ImageLayer::~ImageLayer() {}

void ImageLayer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InputLayer::initialize(name, params, comm);
}

ActivityComponent *ImageLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
