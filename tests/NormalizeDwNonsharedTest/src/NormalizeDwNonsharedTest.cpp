/*
 * NormalizeDwNonsharedTest.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <components/WeightsPair.hpp>
#include <connections/HyPerConn.hpp>
#include <structures/WeightData.hpp>

#include <cassert>
#include <memory>

int checkWeights(HyPerCol *hc, int argc, char **argv);
std::shared_ptr<WeightData> findWeightData(HyPerCol *hc, std::string const &connName);

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, checkWeights /*customexit*/);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkWeights(HyPerCol *hc, int argc, char **argv) {
   float const relativeTolerance = 1.2e-6f;
   // This is the maximum allowed relative error between the correct values and the calculated
   // values. It's hard-coded, which should be changed at some point. The choice of values is
   // somewhat pessimistic (hopefully). The relative error if a single-precision floating point
   // value is off by one ULP is at most 1.1921e-7. There are five updates to the weights and
   // in each one, the delta-weights are normalized and then added to the existing weights.
   // Crudely, we allow for both the dw-normalization and the addition to be off by 1 ULP each.
   // Again crudely, we allow for a worst-case scenario of an error of 10 ULP.
   // It's almost certainly possible to choose a sharper tolerance with a more sophitsticated
   // analysis. But this tolerance is small enough that it we expect it won't be be exceeded
   // unless there's a bug.

   auto fileManager = std::make_shared<FileManager>(
         hc->getCommunicator()->getIOMPIBlock(), "output");
   auto observedWeightData = findWeightData(hc, "InputToOutput");
   auto correctWeightData = findWeightData(hc, "CorrectWeights");

   FatalIf(
         observedWeightData->getNumArbors() != 1,
         "InputToOutput has NumArbors=%d but there should only be a single arbor for this test.",
         observedWeightData->getNumArbors());
   FatalIf(
         correctWeightData->getNumArbors() != 1,
         "CorrectWeights has NumArbors=%d but there should only be a single arbor for this test.",
         correctWeightData->getNumArbors());
   int observedPatchSizeOverall = observedWeightData->getPatchSizeOverall();
   int correctPatchSizeOverall = correctWeightData->getPatchSizeOverall();
   FatalIf(
         observedPatchSizeOverall != correctPatchSizeOverall,
         "InputToOutput has overall patch size %d but CorrectWeights has %d\n",
         observedWeightData->getPatchSizeOverall(), correctWeightData->getPatchSizeOverall());
   int observedNumDataPatches = observedWeightData->getNumDataPatchesOverall();
   int correctNumDataPatches = correctWeightData->getNumDataPatchesOverall();
   FatalIf(
         observedNumDataPatches != correctNumDataPatches,
         "InputToOutput has overall patch size %d but CorrectWeights has %d\n",
         observedWeightData->getPatchSizeOverall(), correctWeightData->getPatchSizeOverall());

   int numWeightValues = correctPatchSizeOverall * correctNumDataPatches;
   assert(observedPatchSizeOverall * observedNumDataPatches == numWeightValues); 
   float const *observed = observedWeightData->getData(0);
   float const *correct  = correctWeightData->getData(0);

   int status = PV_SUCCESS;
   for (int k = 0; k < numWeightValues; ++k) {
      if (observed[k] == correct[k]) { continue; }
      if (correct[k] == 0.0f) {
         assert(observed[k] != 0.0f);
         ErrorLog().printf(
               "Index %d, correct %f, observed %f, discrepancy %g\n",
               k, double(correct[k]), double(observed[k]), double(observed[k] - correct[k]));
         status = PV_FAILURE;
      }
      else {
         float relerr = (observed[k] - correct[k]) / correct[k];
         if (std::fabs(relerr) >= relativeTolerance) {
         ErrorLog().printf(
               "Index %d, correct %f, observed %f, relative error %g\n",
               k, double(correct[k]), double(observed[k]), double(relerr));
            status = PV_FAILURE;
         }
      }
   }

   return status;
}

std::shared_ptr<WeightData> findWeightData(HyPerCol *hc, std::string const &connName) {
   HyPerConn *conn = dynamic_cast<HyPerConn *>(hc->getObjectFromName(connName));
   FatalIf(conn == nullptr, "No connection named \"%s\" in column.\n", connName.c_str());
   WeightsPair *weightsPair = conn->getComponentByType<WeightsPair>();
   FatalIf(
         weightsPair == nullptr,
         "Connection \"%s\" does not have a WeightsPair component.\n", connName.c_str());
   Weights *weights = weightsPair->getPreWeights();
   FatalIf(
         weights == nullptr,
         "Connection \"%s\" does not have pre-perspective weights.\n", connName.c_str());
   return weights->getData();
}
