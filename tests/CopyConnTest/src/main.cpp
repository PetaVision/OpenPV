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
   int rank         = 0;
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

   HyPerConn *origConn = dynamic_cast<HyPerConn *>(hc->getObjectFromName("OriginalConn"));
   if (origConn == NULL) {
      if (rank == 0) {
         ErrorLog().printf(
               "Unable to find connection named \"OriginalConn\" in params file \"%s\".\n",
               paramsfile);
      }
      status = PV_FAILURE;
   }
   CopyConn *copyConn = dynamic_cast<CopyConn *>(hc->getObjectFromName("CopyConn"));
   if (copyConn == NULL) {
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

   float origStrength = origConn->getStrength();

   float copyStrength = copyConn->getStrength();

   int origNumPatches = origConn->getNumDataPatches();
   int copyNumPatches = copyConn->getNumDataPatches();
   FatalIf(origNumPatches != copyNumPatches, "Test failed.\n");
   int origNxp = origConn->getPatchSizeX();
   int copyNxp = copyConn->getPatchSizeX();
   FatalIf(origNxp != copyNxp, "Test failed.\n");
   int origNyp = origConn->getPatchSizeY();
   int copyNyp = copyConn->getPatchSizeY();
   FatalIf(origNyp != copyNyp, "Test failed.\n");
   int origNfp = origConn->getPatchSizeF();
   int copyNfp = copyConn->getPatchSizeF();
   FatalIf(origNfp != copyNfp, "Test failed.\n");
   int origNumArbors = origConn->getNumAxonalArbors();
   int copyNumArbors = copyConn->getNumAxonalArbors();
   FatalIf(origNumArbors != copyNumArbors, "Test failed.\n");

   for (int arbor = 0; arbor < origNumArbors; arbor++) {
      for (int patchindex = 0; patchindex < origNumPatches; patchindex++) {
         for (int y = 0; y < origNyp; y++) {
            for (int x = 0; x < origNxp; x++) {
               for (int f = 0; f < origNfp; f++) {
                  int indexinpatch = kIndex(x, y, f, origNxp, origNyp, origNfp);
                  float origWeight = origConn->getWeightsDataHead(arbor, patchindex)[indexinpatch];
                  float copyWeight = copyConn->getWeightsDataHead(arbor, patchindex)[indexinpatch];
                  float discrep    = fabsf(origWeight * copyStrength - copyWeight * origStrength);
                  if (discrep > 1e-6f) {
                     ErrorLog().printf(
                           "Rank %d: arbor %d, patchindex %d, x=%d, y=%d, f=%d: discrepancy of "
                           "%g\n",
                           hc->columnId(),
                           arbor,
                           patchindex,
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
