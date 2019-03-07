#include "ImagePvpOffsetTestLayer.hpp"

#include "ImagePvpOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpOffsetTestLayer::ImagePvpOffsetTestLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

ImagePvpOffsetTestLayer::~ImagePvpOffsetTestLayer() {}

void ImagePvpOffsetTestLayer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   PvpLayer::initialize(name, params, comm);
}

ActivityComponent *ImagePvpOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpOffsetTestActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

Response::Status ImagePvpOffsetTestLayer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = PvpLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   FatalIf(
         message->mDeltaTime != 1.0,
         "This test requires the HyPerCol dt parameter equal 1.0 (value is %f).\n",
         message->mDeltaTime);
   return Response::SUCCESS;
}; /* class ImageOffsetTestLayer */

} /* namespace PV */
