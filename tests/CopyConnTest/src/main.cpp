/*
 * main .cpp file for CopyConnTest
 *
 */


#include <columns/buildandrun.hpp>
#include <connections/CopyConn.hpp>
#include <connections/HyPerConn.hpp>
#include <normalizers/NormalizeBase.hpp>

int runparamsfile(PV_Init* initObj, char const * paramsfile);

int main(int argc, char * argv[]) {
   int rank = 0;
   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   if (initObj->getParamsFile()) {
      if(initObj->getWorldRank()==0) {
         pvErrorNoExit() << argv[0] << " should be run without the params file argument.\n" <<
               "This test uses several hard-coded params files\n";
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   int status = PV_SUCCESS;

   if (status == PV_SUCCESS) { status = runparamsfile(initObj, "input/CopyConnInitializeTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(initObj, "input/CopyConnInitializeNonsharedTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(initObj, "input/CopyConnPlasticTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(initObj, "input/CopyConnPlasticNonsharedTest.params"); }

   delete initObj;
   return status;
}

// Given one params file, runparamsfile builds and runs, but then before deleting the HyPerCol,
// it looks for a connection named "OriginalConn" and one named "CopyConn",
// grabs the normalization strengths of each, and tests whether the weights divided by the strength
// are equal to within roundoff error.
// (Technically, it tests whether (original weight)*(copy strength) and (copy weight)*(original strength)
// are within 1.0e-6 in absolute value.  This is reasonable if the weights and strengths are order-of-magnitude 1.0)
// 
// Note that this check makes assumptions on the normalization method, although normalizeSum, normalizeL2 and normalizeMax all satisfy them.
int runparamsfile(PV_Init* initObj, char const * paramsfile) {
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int status = initObj->setParams(paramsfile);
   pvErrorIf(status!=PV_SUCCESS, "Test failed.\n");

   HyPerCol * hc = build(initObj);
   if (hc != NULL) {
      status = hc->run();
      if( status != PV_SUCCESS ) {
         if (rank==0) {
            pvErrorNoExit().printf("%s: running with params file %s returned status code %d.\n",
                  initObj->getProgramName(), paramsfile, status);
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

   HyPerConn * origConn = dynamic_cast<HyPerConn *>(hc->getConnFromName("OriginalConn"));
   if (origConn==NULL) {
      if (rank==0) {
         pvErrorNoExit().printf("Unable to find connection named \"OriginalConn\" in params file \"%s\".\n", paramsfile);
      }
      status = PV_FAILURE;
   }
   CopyConn * copyConn = dynamic_cast<CopyConn *>(hc->getConnFromName("CopyConn"));
   if (origConn==NULL) {
      if (rank==0) {
         pvErrorNoExit().printf("Unable to find connection named \"CopyConn\" in params file \"%s\".\n", paramsfile);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      delete hc;
      return status;
   }

   NormalizeBase * origNormalizer = origConn->getNormalizer();
   pvErrorIf(!(origNormalizer), "Test failed.\n");
   float origStrength = origNormalizer->getStrength();

   NormalizeBase * copyNormalizer = copyConn->getNormalizer();
   pvErrorIf(!(copyNormalizer), "Test failed.\n");
   float copyStrength = copyNormalizer->getStrength();

   int origNumPatches = origConn->getNumDataPatches();
   int copyNumPatches = copyConn->getNumDataPatches();
   pvErrorIf(!(origNumPatches==copyNumPatches), "Test failed.\n");
   int origNxp = origConn->xPatchSize();
   int copyNxp = copyConn->xPatchSize();
   pvErrorIf(!(origNxp==copyNxp), "Test failed.\n");
   int origNyp = origConn->yPatchSize();
   int copyNyp = copyConn->yPatchSize();
   pvErrorIf(!(origNyp==copyNyp), "Test failed.\n");
   int origNfp = origConn->fPatchSize();
   int copyNfp = copyConn->fPatchSize();
   pvErrorIf(!(origNfp==copyNfp), "Test failed.\n");
   int origNumArbors = origConn->numberOfAxonalArborLists();
   int copyNumArbors = copyConn->numberOfAxonalArborLists();
   pvErrorIf(!(origNumArbors==copyNumArbors), "Test failed.\n");

   for (int arbor=0; arbor<origNumArbors; arbor++) {
      for (int patchindex=0; patchindex<origNumPatches; patchindex++) {
         for (int y=0; y<origNyp; y++) {
            for (int x=0; x<origNxp; x++) {
               for (int f=0; f<origNfp; f++) {
                  int indexinpatch = kIndex(x,y,f,origNxp,origNyp,origNfp);
                  pvwdata_t origWeight = origConn->get_wDataHead(arbor, patchindex)[indexinpatch];
                  pvwdata_t copyWeight = copyConn->get_wDataHead(arbor, patchindex)[indexinpatch];
                  float discrep = fabsf(origWeight*copyStrength - copyWeight*origStrength);
                  if (discrep > 1e-6) {
                     pvErrorNoExit().printf("Rank %d: arbor %d, patchindex %d, x=%d, y=%d, f=%d: discrepancy of %g\n",
                           hc->columnId(), arbor, patchindex, x, y, f, discrep);
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
