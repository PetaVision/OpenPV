#include "ImageCollationLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/ImageCollationActivityBuffer.hpp"

namespace PV {

ImageCollationLayer::ImageCollationLayer(
      std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

ImageCollationLayer::~ImageCollationLayer() {}

void ImageCollationLayer::initialize(
      std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   InputLayer::initialize(paramsIO, comm);
}

ActivityComponent *ImageCollationLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageCollationActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
