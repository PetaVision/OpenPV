#include "ImagePvpTestLayer.hpp"
#include "ImagePvpTestBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpTestLayer::ImagePvpTestLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

ImagePvpTestLayer::~ImagePvpTestLayer() {}

void ImagePvpTestLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   InputLayer::initialize(params, defaults, comm);
}

ActivityComponent *ImagePvpTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpTestBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
