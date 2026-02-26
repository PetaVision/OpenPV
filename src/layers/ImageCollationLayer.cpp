#include "ImageCollationLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/ImageCollationActivityBuffer.hpp"

namespace PV {

ImageCollationLayer::ImageCollationLayer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ImageCollationLayer::~ImageCollationLayer() {}

void ImageCollationLayer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InputLayer::initialize(name, params, comm);
}

ActivityComponent *ImageCollationLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageCollationActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
