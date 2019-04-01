/*
 * main .cpp file for CopyConnTest
 *
 */

#include <columns/buildandrun.hpp>
#include <connections/CopyConn.hpp>
#include <connections/HyPerConn.hpp>
#include <normalizers/NormalizeBase.hpp>

int runparamsfile(PV_Init *initObj, char const *paramsfile);

int main(int argc, char *argv[]) {
   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   if (initObj->getParams()) {
      if (initObj->getWorldRank() == 0) {
         ErrorLog() << argv[0] << " should be run without the params file argument.\n"
                    << "This test uses several hard-coded params files\n";
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   int status = PV_SUCCESS;

   if (status == PV_SUCCESS) {
      status = runparamsfile(initObj, "input/CopyConnInitializeTest.params");
   }
   if (status == PV_SUCCESS) {
      status = runparamsfile(initObj, "input/CopyConnInitializeNonsharedTest.params");
   }
   if (status == PV_SUCCESS) {
      status = runparamsfile(initObj, "input/CopyConnPlasticTest.params");
   }
   if (status == PV_SUCCESS) {
      status = runparamsfile(initObj, "input/CopyConnPlasticNonsharedTest.params");
   }

   delete initObj;
   return status;
}

// Given one params file, runparamsfile builds and runs, but then before deleting the HyPerCol,
// it looks for a connection named "OriginalConn" and one named "CopyConn",
// grabs the normalization strengths of each, and tests whether the weights divided by the strength
// are equal to within roundoff error.
// (Technically, it tests whether (original weight)*(copy strength) and (copy weight)*(original
// strength)
// are within 1.0e-6 in absolute value.  This is reasonable if the weights and strengths are
// order-of-magnitude 1.0)
//
// Note that this check makes assumptions on the normalization method, although normalizeSum,
// normalizeL2 and normalizeMax all satisfy them.
int runparamsfile(PV_Init *initObj, char const *paramsfile) {
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int status = initObj->setParams(paramsfile);
   FatalIf(status != PV_SUCCESS, "Test failed.\n");

   HyPerCol *hc = build(initObj);
   if (hc != NULL) {
      status = hc->run();
      if (status != PV_SUCCESS) {
         if (rank == 0) {
            ErrorLog().printf(
                  "%s: running with params file %s returned status code %d.\n",
                  initObj->getProgramName(),
                  paramsfile,
                  status);
         }
      }
   }
   else {
      status = PV_FAILURE;
   }

   if (status != PV_SUCCESS) {
      delete hc;
      return status;
   }

   auto *origConn = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("OriginalConn"));
   if (origConn == nullptr) {
      if (rank == 0) {
         ErrorLog().printf(
               "Unable to find connection named \"OriginalConn\" in params file \"%s\".\n",
               paramsfile);
      }
      status = PV_FAILURE;
   }
   auto *copyConn = dynamic_cast<ComponentBasedObject *>(hc->getObjectFromName("CopyConn"));
   if (copyConn == nullptr) {
      if (rank == 0) {
         ErrorLog().printf(
               "Unable to find connection named \"CopyConn\" in params file \"%s\".\n", paramsfile);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      delete hc;
      return status;
   }

   float origStrength = origConn->getComponentByType<NormalizeBase>()->getStrength();
   float copyStrength = copyConn->getComponentByType<NormalizeBase>()->getStrength();

   int origNumPatches =
         origConn->getComponentByType<WeightsPair>()->getPreWeights()->getNumDataPatches();
   int copyNumPatches =
         copyConn->getComponentByType<WeightsPair>()->getPreWeights()->getNumDataPatches();
   FatalIf(
         origNumPatches != copyNumPatches,
         "Test failed. OriginalConn has %d patches but CopyConn has %d.\n",
         origNumPatches,
         copyNumPatches);

   auto *origPatchSize = origConn->getComponentByType<PatchSize>();
   int origNxp         = origPatchSize->getPatchSizeX();
   int origNyp         = origPatchSize->getPatchSizeY();
   int origNfp         = origPatchSize->getPatchSizeF();

   auto *copyPatchSize = copyConn->getComponentByType<PatchSize>();
   int copyNxp         = copyPatchSize->getPatchSizeX();
   int copyNyp         = copyPatchSize->getPatchSizeY();
   int copyNfp         = copyPatchSize->getPatchSizeF();
   FatalIf(
         origNxp != copyNxp || origNyp != copyNyp || origNfp != copyNfp,
         "Test failed. OriginalConn has patchsize %dx%dx%d but CopyConn has %dx%dx%d.\n",
         origNxp,
         origNyp,
         origNfp,
         copyNxp,
         copyNyp,
         copyNfp);

   int origNumArbors = origConn->getComponentByType<ArborList>()->getNumAxonalArbors();
   int copyNumArbors = copyConn->getComponentByType<ArborList>()->getNumAxonalArbors();
   FatalIf(
         origNumArbors != copyNumArbors,
         "Test failed. OriginalConn has %d arbors, but CopyConn has %d.\n",
         origNumArbors,
         copyNumArbors);

   auto *origPreWeights = origConn->getComponentByType<WeightsPair>()->getPreWeights();
   auto *copyPreWeights = copyConn->getComponentByType<WeightsPair>()->getPreWeights();
   for (int arbor = 0; arbor < origNumArbors; arbor++) {
      for (int patchIndex = 0; patchIndex < origNumPatches; patchIndex++) {
         float *origWeightsData = origPreWeights->getDataFromDataIndex(arbor, patchIndex);
         float *copyWeightsData = copyPreWeights->getDataFromDataIndex(arbor, patchIndex);
         for (int y = 0; y < origNyp; y++) {
            for (int x = 0; x < origNxp; x++) {
               for (int f = 0; f < origNfp; f++) {
                  int indexinpatch = kIndex(x, y, f, origNxp, origNyp, origNfp);
                  float origWeight = origWeightsData[indexinpatch];
                  float copyWeight = copyWeightsData[indexinpatch];
                  float discrep    = fabsf(origWeight * copyStrength - copyWeight * origStrength);
                  if (discrep > 1e-6f) {
                     ErrorLog().printf(
                           "Rank %d: arbor %d, patchIndex %d, x=%d, y=%d, f=%d: discrepancy of "
                           "%g\n",
                           hc->columnId(),
                           arbor,
                           patchIndex,
                           x,
                           y,
                           f,
                           (double)discrep);
                     status = PV_FAILURE;
                  }
               }
            }
         }
      }
   }

   delete hc;

   return status;
}
