#include "ImagePvpTestLayer.hpp"
#include "ImagePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpTestLayer::ImagePvpTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

ImagePvpTestLayer::~ImagePvpTestLayer() {}

void ImagePvpTestLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   InputLayer::initialize(paramsIO, comm);
}

ActivityComponent *ImagePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpTestBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
