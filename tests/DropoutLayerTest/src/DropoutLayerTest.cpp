#include <columns/buildandrun.hpp>
#include <layers/HyPerLayer.hpp>

/*
   Tests dropout by sending a ConstantLayer full of ones
   through a DropoutLayer and averaging the values that
   come out the other side.
*/

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status;

   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);

   /* First param file has 5% dropout. From there, it's 25%, 50%, 75%, and 95% */

   initObj.setParams("input/DropoutLayerTest_05.params");
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_25.params");
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_50.params");
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_75.params");
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   initObj.setParams("input/DropoutLayerTest_95.params");
   status    = rebuildandrun(&initObj, NULL, &customexit);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   char *programPathC = strdup(argv[0]);
   char *programNameC = basename(programPathC);
   std::string programName(programNameC);
   free(programPathC);

   int probability;
   hc->parameters()->ioParamValueRequired(PARAMS_IO_READ, "Output", "probability", &probability);
   float targetAvg = (float)(100 - probability) * 0.01f;

   HyPerLayer *averageLayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("Counts"));
   FatalIf(averageLayer == nullptr, "No layer named \"Counts\"\n");
   auto *averagePublisher = averageLayer->getComponentByType<BasePublisherComponent>();
   FatalIf(
         averagePublisher == nullptr, "Layer \"Counts\" does not have a BasePublisherComponent\n");
   float const *checkData = averagePublisher->getLayerData();
   int const numNeurons   = averagePublisher->getNumExtended();

   // Leaky integrator has integration time infinity;
   // Each neuron in the Count layer is the sum over time of the
   // corresponding neuron in the DropoutLayer -- calculate sum over whole layer
   float count = 0.0f;
   for (int k = 0; k < numNeurons; k++) {
      count += checkData[k];
   }

   long int const numTimeSteps = hc->getFinalStep();
   long int const numTrials = numTimeSteps * (long int)numNeurons;
   
   float observedAvg = count / (float)numTrials;

   // count has a binomial distribution with (numNeurons * (number of simulation steps)) trials
   // and success probability of (100-probability)/100 == targetAvg.
   // Thus observedAvg has expected value of targetAvg,
   // and a variance of targetAvg * (1-targetAvg) / numTrials.
   float stddev = std::sqrt(targetAvg * (1-targetAvg) / numTrials);
   float tolerance = 2.0f * stddev;

   FatalIf(
         std::fabs(observedAvg-targetAvg) > tolerance,
         "%s failed: expected average %.5f, observed average %.5f, allowed tolerance %.5f\n",
         programName.c_str(),
         (double)targetAvg,
         (double)observedAvg,
         (double)tolerance);

   if (hc->columnId() == 0) {
      InfoLog().printf(
            "%s passed for dropout probability %2d: expected average %.5f, observed average %.5f).\n",
            programName.c_str(),
            probability,
            (double)targetAvg,
            (double)observedAvg);
   }
   return PV_SUCCESS;
}
