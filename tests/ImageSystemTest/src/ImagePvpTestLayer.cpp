#include "ImagePvpTestLayer.hpp"
#include "ImagePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpTestLayer::ImagePvpTestLayer(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ImagePvpTestLayer::~ImagePvpTestLayer() {}

void ImagePvpTestLayer::initialize(char const *name, PVParams *params, Communicator *comm) {
   InputLayer::initialize(name, params, comm);
}

ActivityComponent *ImagePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
