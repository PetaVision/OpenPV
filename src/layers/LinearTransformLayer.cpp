#include "LinearTransformLayer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/RotateActivityBuffer.hpp"
#include "components/ScaleXActivityBuffer.hpp"
#include "components/ScaleYActivityBuffer.hpp"

namespace PV {

LinearTransformLayer::LinearTransformLayer(
      const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void LinearTransformLayer::initialize(
      const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *LinearTransformLayer::createActivityComponent() {
   std::string const &groupKeyword = parameters()->groupKeywordFromName(getName());

   if (groupKeyword == "RotateLayer") {
      return new HyPerActivityComponent<
            GSynAccumulator,
            HyPerInternalStateBuffer,
            RotateActivityBuffer>(getName(), parameters(), mCommunicator);
   }
   if (groupKeyword == "ScaleXLayer") {
      return new HyPerActivityComponent<
            GSynAccumulator,
            HyPerInternalStateBuffer,
            ScaleXActivityBuffer>(getName(), parameters(), mCommunicator);
   }
   if (groupKeyword == "ScaleYLayer") {
      return new HyPerActivityComponent<
            GSynAccumulator,
            HyPerInternalStateBuffer,
            ScaleYActivityBuffer>(getName(), parameters(), mCommunicator);
   }
   Fatal().printf(
         "LinearTransformLayer \"%s\" has unrecognized group keyword \"%s\"\n",
         getName(), groupKeyword.c_str());
   return nullptr; // never executed because of Fatal(); included to suppress compiler warning
}

} // namespace PV
