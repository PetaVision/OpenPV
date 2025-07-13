#include "ImageLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/ImageActivityBuffer.hpp"

namespace PV {

ImageLayer::ImageLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

ImageLayer::~ImageLayer() {}

void ImageLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InputLayer::initialize(params, defaults, comm);
}

ActivityComponent *ImageLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
