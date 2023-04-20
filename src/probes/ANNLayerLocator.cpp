#include "ANNLayerLocator.hpp"
#include "probes/TargetLayerComponent.hpp"

namespace PV {

ANNActivityBuffer const *
locateANNActivityBuffer(std::shared_ptr<TargetLayerComponent> targetLayerComponent) {
   HyPerLayer *targetLayer = targetLayerComponent->getTargetLayer();
   if (targetLayer == nullptr) {
      ErrorLog().printf(
            "targetLayerComponent \"%s\" target layer is null\n",
            targetLayerComponent->getName_c());
      return nullptr;
   }

   auto *activityComponent = targetLayer->getComponentByType<ActivityComponent>();
   if (activityComponent == nullptr) {
      ErrorLog().printf(
            "target layer \%s\" does not have an activity component\n", targetLayer->getName());
      return nullptr;
   }
   ANNActivityBuffer *activityBuffer = activityComponent->getComponentByType<ANNActivityBuffer>();
   if (activityBuffer == nullptr) {
      ErrorLog().printf(
            "target layer \"%s\" does not have an ANNActivityBuffer component.\n",
            targetLayer->getName());
      return nullptr;
   }
   if (activityBuffer->usingVerticesListInParams() == true) {
      ErrorLog().printf(
         "LCAProbes require target layer \"%s\" to use VThresh etc. "
         "instead of verticesV/verticesA.\n",
         targetLayer->getName());
      return nullptr;
   }
   return activityBuffer;
}

} // namespace PV
