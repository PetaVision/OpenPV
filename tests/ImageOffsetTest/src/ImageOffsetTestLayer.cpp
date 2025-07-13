#include "ImageOffsetTestLayer.hpp"

#include "ImageOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImageOffsetTestLayer::ImageOffsetTestLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

ImageOffsetTestLayer::~ImageOffsetTestLayer() {}

void ImageOffsetTestLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ImageLayer::initialize(params, defaults, comm);
}

ActivityComponent *ImageOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageOffsetTestActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

Response::Status ImageOffsetTestLayer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ImageLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   FatalIf(
         message->mDeltaTime != 1.0,
         "This test requires the HyPerCol dt parameter equal 1.0 (value is %f).\n",
         message->mDeltaTime);
   return Response::SUCCESS;
}

} /* namespace PV */
