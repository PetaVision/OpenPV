#include "ImageLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/ImageActivityBuffer.hpp"

namespace PV {

ImageLayer::ImageLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

ImageLayer::~ImageLayer() {}

void ImageLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   InputLayer::initialize(paramsIO, comm);
}

ActivityComponent *ImageLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
