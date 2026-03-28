/*
 * main.cpp
 *
 */

#include <cassert>
#include <columns/buildandrun.hpp>
#include <components/WeightsPair.hpp>
#include <connections/HyPerConn.hpp>
#include <structures/Weights.hpp>
#include <structures/WeightData.hpp>

using namespace PV;

/* Looks for the presynaptic weights of the connections "LayerAToBroadcastLayerB" and
   "LayerAToBroadcastLayerC", and checks if they are equal. Returns PV_SUCCESS if they
   are equal and PV_FAILURE if not.*/
int checkWeights(HyPerCol *hc, int argc, char **argv);

/* Returns a pointer to the presynaptic weights of the indicated connection */
Weights *findWeights(std::string const &connectionName, HyPerCol const *hc);

class WeightComparison {
  public:
   WeightComparison(
         Weights *weights1, Weights *weights2) : mWeights1(weights1), mWeights2(weights2) {}
   WeightComparison() = delete;
   ~WeightComparison() {}

   template <typename T>
   int testEqual(
         T const &value1, T const &value2, std::string const &valueDescription, int oldStatus);

  private:
   Weights const *mWeights1;
   Weights const *mWeights2;
};

template <typename T>
int WeightComparison::testEqual(
      T const &value1, T const &value2, std::string const &valueDescription, int oldStatus) {
   if (value1 != value2) {
      ErrorLog() << "\"" << mWeights1->getName() << "\" and \"" << mWeights2->getName()
                 << "\" have different \"" << valueDescription << "\" values ("
                 << value1 << " versus " << value2 << ".\n";
      return PV_FAILURE;
   }
   return oldStatus;
}

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv, nullptr /*init hook*/, checkWeights /*exit hook*/);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkWeights(HyPerCol *hc, int argc, char **argv) {
   Weights *originalWeights = findWeights("LayerAToBroadcastLayerB", hc);
   assert(originalWeights != nullptr); // findWeights errors out instead of returning null.
   Weights *transposeOfTranspose = findWeights("LayerAToBroadcastLayerC", hc);
   assert(transposeOfTranspose != nullptr); // findWeights errors out instead of returning null.

   int status = PV_SUCCESS;
   WeightComparison weightComparison(originalWeights, transposeOfTranspose);
   status = weightComparison.testEqual(
         originalWeights->getSharedWeightsFlag(),
         transposeOfTranspose->getSharedWeightsFlag(),
         "SharedWeightsFlag",
         status);

   status = weightComparison.testEqual(
         originalWeights->prelayerIsBroadcast(),
         transposeOfTranspose->prelayerIsBroadcast(),
         "PreLayerIsBroadcast",
         status);

   status = weightComparison.testEqual(
         originalWeights->postlayerIsBroadcast(),
         transposeOfTranspose->postlayerIsBroadcast(),
         "PostLayerIsBroadcast",
         status);

   auto originalWeightData = originalWeights->getData();
   auto transposeOfTransposeData = transposeOfTranspose->getData();

   status = weightComparison.testEqual(
         originalWeightData->getNumDataPatchesX(),
         transposeOfTransposeData->getNumDataPatchesX(),
         "NumDataPatchesX",
         status);

   status = weightComparison.testEqual(
         originalWeightData->getNumDataPatchesY(),
         transposeOfTransposeData->getNumDataPatchesY(),
         "NumDataPatchesX",
         status);

   status = weightComparison.testEqual(
         originalWeightData->getPatchSizeX(),
         transposeOfTransposeData->getPatchSizeX(),
         "NumDataPatchesX",
         status);

   status = weightComparison.testEqual(
         originalWeightData->getPatchSizeY(),
         transposeOfTransposeData->getPatchSizeY(),
         "NumDataPatchesX",
         status);

   status = weightComparison.testEqual(
         originalWeightData->getPatchSizeF(),
         transposeOfTransposeData->getPatchSizeF(),
         "NumDataPatchesX",
         status);

   FatalIf(status != PV_SUCCESS, "Test failed.\n");

   assert(originalWeightData->getNumArbors() == 1);
   assert(transposeOfTransposeData->getNumArbors() == 1);

   float const *originalDataPtr = originalWeightData->getData(0);
   float const *transposeOfTransposeDataPtr = transposeOfTransposeData->getData(0);

   long int N = originalWeightData->getNumValuesPerArbor();
   assert(transposeOfTransposeData->getNumValuesPerArbor() == N);
   // previous testEqual() calls prove NumValuesPerArbor will the the same.

   for (long int n = 0; n < N; ++n) {
      float original = originalDataPtr[n];
      float transposeOfTranspose = transposeOfTransposeDataPtr[n];
      if (transposeOfTranspose != original) {
         ErrorLog().printf(
               "Weight discrepancy at index %ld: %f versus %f (discrepancy %g)\n",
               n, double(original), double(transposeOfTranspose),
               double(transposeOfTranspose - original));
      }
   }

   return status;
}

Weights *findWeights(std::string const &name, HyPerCol const *hc) {
   HyPerConn *conn = dynamic_cast<HyPerConn *>(hc->getObjectFromName(name));
   FatalIf(conn == nullptr, "Unable to find connection \"%s\"\n", name.c_str());
   auto *weightsPair = conn->getComponentByType<WeightsPair>();
   FatalIf(weightsPair == nullptr, "Connection \"%s\" has no WeightsPair.\n", name.c_str());
   Weights *weights = weightsPair->getPreWeights();
   FatalIf(weights == nullptr, "Connection \"%s\" has no presynaptic weights.\n", name.c_str());
   return weights;
}
