/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <connections/HyPerConn.hpp>

int customexit(HyPerCol *hc, int argc, char **argv);
void checkWeights(HyPerCol *hc, std::string const &objName, float correctWeight);

int main(int argc, char *argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char **argv) {
   checkWeights(hc, std::string("initializeFromInitWeights"), 1.0f);
   checkWeights(hc, std::string("initializeFromCheckpoint"), 2.0f);
   return PV_SUCCESS;
}

void checkWeights(HyPerCol *hc, std::string const &objName, float correctWeight) {
   auto *conn = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName(objName.c_str()));
   FatalIf(
         conn == nullptr,
         "Test failed. No object named \"%s\" in the hierarchy.\n",
         objName.c_str());

   auto *patchSize = conn->getComponentByType<PatchSize>();
   FatalIf(
         patchSize == nullptr,
         "Test failed. %s does not have a PatchSize component.\n",
         conn->getDescription_c());
   FatalIf(
         patchSize->getPatchSizeX() != 1 || patchSize->getPatchSizeY() != 1
               || patchSize->getPatchSizeF() != 1,
         "Test failed. Connection \"%s\" must have patch size 1x1x1.\n",
         conn->getDescription_c());

   auto *arborList = conn->getComponentByType<ArborList>();
   FatalIf(
         arborList == nullptr,
         "Test failed. %s does not have an ArborList component.\n",
         conn->getDescription_c());
   FatalIf(
         arborList->getNumAxonalArbors() != 1,
         "Test failed. Connection %s has %d arbors, instead of one.\n",
         conn->getDescription_c(),
         arborList->getNumAxonalArbors());

   auto *weightsPair = conn->getComponentByType<WeightsPair>();
   FatalIf(
         weightsPair == nullptr,
         "Test failed. %s does not have a WeightsPair component.\n",
         conn->getDescription_c());
   auto *preWeights     = weightsPair->getPreWeights();
   float observedWeight = preWeights->getDataFromDataIndex(0, 0)[0];
   FatalIf(
         observedWeight != correctWeight,
         "Test failed. Connection %s had weight %f; should be %f.\n",
         conn->getDescription_c(),
         (double)observedWeight,
         (double)correctWeight);
}
