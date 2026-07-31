/*
 * pv.cpp
 *
 */

#include <columns/ComponentBasedObject.hpp>
#include <columns/buildandrun.hpp>
#include <components/BasePublisherComponent.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   if (initObj.getParams() == nullptr) {
      initObj.setParams("input/CloneVLayerTest.params");
   }

   int status;
   status = rebuildandrun(&initObj, NULL, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   auto *checkClone = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("CheckClone"));
   FatalIf(checkClone == nullptr, "No layer named \"CheckClone\"\n");
   auto *checkClonePublisher = checkClone->getComponentByType<BasePublisherComponent>();
   FatalIf(
         checkClonePublisher == nullptr,
         "Layer \"CheckClone\" does not have a BasePublisherComponent\n");
   float const *checkCloneLayerData = checkClonePublisher->getLayerData();
   long const numCloneLayerNeurons  = checkClonePublisher->getNumExtended();
   for (long k = 0; k < numCloneLayerNeurons; k++) {
      FatalIf(std::fabs(checkCloneLayerData[k]) >= 1.0e-6f, "Test failed.\n");
   }

   auto *checkSigmoid = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("CheckSigmoid"));
   FatalIf(checkSigmoid == nullptr, "No layer named \"CheckSigmoid\"\n");
   auto *checkSigmoidPublisher = checkSigmoid->getComponentByType<BasePublisherComponent>();
   FatalIf(
         checkSigmoidPublisher == nullptr,
         "Layer \"CheckSigmoid\" does not have a BasePublisherComponent\n");
   float const *checkSigmoidLayerData = checkSigmoidPublisher->getLayerData();
   long const numSigmoidLayerNeurons  = checkSigmoidPublisher->getNumExtended();
   for (long k = 0; k < numSigmoidLayerNeurons; k++) {
      FatalIf(std::fabs(checkSigmoidLayerData[k]) >= 1.0e-6f, "Test failed.\n");
   }

   if (hc->columnId() == 0) {
      InfoLog().printf("%s passed.\n", argv[0]);
   }
   return PV_SUCCESS;
}
