#include <columns/buildandrun.hpp>
#include <layers/HyPerLayer.hpp>

/*
   Tests dropout by sending a ConstantLayer full of ones
   through a DropoutLayer and averaging the values that
   come out the other side.
*/

static float targetAvg;

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status;

   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);

   /* First param file has 5% dropout. From there, it's 25%, 50%, 75%, and 95% */

   initObj.setParams("input/DropoutLayerTest_05.params");
   targetAvg = 95.0f;
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_25.params");
   targetAvg = 75.0f;
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_50.params");
   targetAvg = 50.0f;
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_75.params");
   targetAvg = 25.0f;
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_95.params");
   targetAvg = 5.0f;
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   HyPerLayer *averageLayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("Average"));
   FatalIf(averageLayer == nullptr, "No layer named \"Average\"\n");
   auto *averagePublisher = averageLayer->getComponentByType<BasePublisherComponent>();
   FatalIf(
         averagePublisher == nullptr, "Layer \"Average\" does not have a BasePublisherComponent\n");
   float const *checkData = averagePublisher->getLayerData();
   int const numNeurons   = averagePublisher->getNumExtended();

   /* Leaky integrator has average per neuron- calculate average over whole layer */
   float count = 0.0f;
   for (int k = 0; k < numNeurons; k++) {
      count += checkData[k];
   }

   float observedAvg = count / (float)numNeurons;
   FatalIf(
         fabsf(observedAvg - targetAvg) > 1,
         "Test failed: expected average %f, observed average %f\n",
         (double)targetAvg, (double)observedAvg);

   if (hc->columnId() == 0) {
      InfoLog().printf("%s passed for dropout probability %.1f%%.\n", argv[0], 100.0 - (double)targetAvg);
   }
   return PV_SUCCESS;
}
