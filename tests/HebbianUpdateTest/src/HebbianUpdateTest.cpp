/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <connections/HyPerConn.hpp>
#include <components/WeightsPair.hpp>

int checkWeights(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr, checkWeights);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkWeights(HyPerCol *hc, int argc, char *argv[]) {
   Observer *observer = hc->getObjectFromName(std::string("PreToPost"));
   HyPerConn *conn = dynamic_cast<HyPerConn*>(observer);
   FatalIf(
         conn == nullptr,
         "Connection \"PreToPost\" not found.\n");
   WeightsPair *weightsPair = conn->getComponentByType<WeightsPair>();
   FatalIf(
         weightsPair == nullptr,
         "Connection \"PreToPost\" does not have a WeightsPair component.\n");
   Weights *weights = weightsPair->getPreWeights();
   FatalIf(
         weights == nullptr,
         "Connection \"PreToPost\" does not have weights with pre-synaptic perspective.\n");
   int status = PV_SUCCESS;
   int nxp = weights->getPatchSizeX();
   if (nxp != 8) {
      ErrorLog().printf("Weights have patch width %d instead of expected 8.\n", nxp);
      status = PV_FAILURE;
   }
   int nyp = weights->getPatchSizeY();
   if (nyp != 8) {
      ErrorLog().printf("Weights have patch height %d instead of expected 8.\n", nyp);
      status = PV_FAILURE;
   }
   int nfp = weights->getPatchSizeF();
   if (nfp != 1) {
      ErrorLog().printf("Weights have feature depth %d instead of expected 1.\n", nfp);
      status = PV_FAILURE;
   }
   long numWeightsPerPatch = weights->getPatchSizeOverall();
   long numKernels = weights->getNumDataPatchesOverall();
   if (numKernels != 8L) {
      ErrorLog().printf("Weights have %ld kernels instead of expected 8.\n", numKernels);
      status = PV_FAILURE;
   }
   std::vector<std::vector<float>> correctValues(2);
   correctValues[0] = std::vector<float>{7.0f, 8.0f, 5.0f, 6.0f};
   correctValues[1] = std::vector<float>{3.0f, 4.0f, 1.0f, 2.0f};
   for (long p = 0L; p < numKernels; ++p) {
      float multiplier = 2.5f * static_cast<float>(p + 1L);
      // dWMax = 0.5 and there were 5 update periods; hence the 2.5 factor.
      // The presynaptic layer was all ones in feature 0, all twos in feature 1, etc.;
      // hence the (p + 1) factor.
      float const *weightValues = weights->getDataFromPatchIndex(0, p);
      for (long w = 0; w < numWeightsPerPatch; ++w) {
         float observed = weightValues[w];
         int cx = kxPos(w, nxp, nyp, nfp) % 4;
         int cy = kyPos(w, nxp, nyp, nfp) % 2;
         float expected = correctValues[cy][cx] * multiplier;
         if (observed != expected) {
            ErrorLog().printf(
                  "Patch %ld, weight at index %ld, expected %f, observed %f\n",
                  p, w, (double)expected, (double)observed);
         }
      }
   }
   return status;
}
