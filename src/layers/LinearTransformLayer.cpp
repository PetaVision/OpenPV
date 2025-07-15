#include "LinearTransformLayer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/RotateActivityBuffer.hpp"
#include "components/ScaleXActivityBuffer.hpp"
#include "components/ScaleYActivityBuffer.hpp"

namespace PV {

LinearTransformLayer::LinearTransformLayer(
      std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

void LinearTransformLayer::initialize(
      std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *LinearTransformLayer::createActivityComponent() {
   std::string const &groupKeyword = getKeyword();

   if (groupKeyword == "RotateLayer") {
      return new HyPerActivityComponent<
            GSynAccumulator,
            HyPerInternalStateBuffer,
            RotateActivityBuffer>(mParamsIO, mCommunicator);
   }
   if (groupKeyword == "ScaleXLayer") {
      return new HyPerActivityComponent<
            GSynAccumulator,
            HyPerInternalStateBuffer,
            ScaleXActivityBuffer>(mParamsIO, mCommunicator);
   }
   if (groupKeyword == "ScaleYLayer") {
      return new HyPerActivityComponent<
            GSynAccumulator,
            HyPerInternalStateBuffer,
            ScaleYActivityBuffer>(mParamsIO, mCommunicator);
   }
   Fatal().printf(
         "LinearTransformLayer \"%s\" has unrecognized group keyword \"%s\"\n",
         getName(), groupKeyword.c_str());
   return nullptr; // never executed because of Fatal(); included to suppress compiler warning
}

} // namespace PV
