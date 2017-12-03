/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <layers/HyPerLayer.hpp>

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
   HyPerLayer *checkCloneLayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("CheckClone"));
   FatalIf(checkCloneLayer == nullptr, "No layer named \"CheckClone\"\n");
   float const *checkCloneLayerData = checkCloneLayer->getLayerData();
   int const numCloneLayerNeurons   = checkCloneLayer->getNumExtended();
   for (int k = 0; k < numCloneLayerNeurons; k++) {
      FatalIf(fabsf(checkCloneLayerData[k]) >= 1.0e-6f, "Test failed.\n");
   }

   HyPerLayer *checkSigmoidLayer =
         dynamic_cast<HyPerLayer *>(hc->getObjectFromName("CheckSigmoid"));
   FatalIf(checkSigmoidLayer == nullptr, "No layer named \"CheckSigmoid\"\n");
   float const *checkSigmoidLayerData = checkSigmoidLayer->getLayerData();
   int const numSigmoidLayerNeurons   = checkSigmoidLayer->getNumExtended();
   for (int k = 0; k < numSigmoidLayerNeurons; k++) {
      FatalIf(fabsf(checkSigmoidLayerData[k]) >= 1.0e-6f, "Test failed.\n");
   }

   if (hc->columnId() == 0) {
      InfoLog().printf("%s passed.\n", argv[0]);
   }
   return PV_SUCCESS;
}
