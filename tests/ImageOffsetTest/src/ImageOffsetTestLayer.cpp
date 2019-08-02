#include "ImageOffsetTestLayer.hpp"

#include "ImageOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImageOffsetTestLayer::ImageOffsetTestLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

ImageOffsetTestLayer::~ImageOffsetTestLayer() {}

void ImageOffsetTestLayer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ImageLayer::initialize(name, params, comm);
}

ActivityComponent *ImageOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImageOffsetTestActivityBuffer>(
         getName(), parameters(), mCommunicator);
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
