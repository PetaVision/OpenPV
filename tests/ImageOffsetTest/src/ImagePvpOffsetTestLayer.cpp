#include "ImagePvpOffsetTestLayer.hpp"

#include "ImagePvpOffsetTestActivityBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

ImagePvpOffsetTestLayer::ImagePvpOffsetTestLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

ImagePvpOffsetTestLayer::~ImagePvpOffsetTestLayer() {}

void ImagePvpOffsetTestLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   PvpLayer::initialize(paramsIO, comm);
}

ActivityComponent *ImagePvpOffsetTestLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<ImagePvpOffsetTestActivityBuffer>(
         mParamsIO, mCommunicator);
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
