/*
 * main .cpp file for CopyConnTest
 *
 */


#include <columns/buildandrun.hpp>
#include <connections/CopyConn.hpp>
#include <connections/HyPerConn.hpp>
#include <normalizers/NormalizeBase.hpp>

int runparamsfile(int argc, char ** argv, PV_Init* initObj, char const * paramsfile);

int main(int argc, char * argv[]) {
   int rank = 0;
   PV_Init* initObj = new PV_Init(&argc, &argv);
//#ifdef PV_USE_MPI
//   MPI_Init(&argc, &argv);
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//#endif // PV_USE_MPI

   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", argv[0]);
         fprintf(stderr, "This test uses several hard-coded params files\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
    }

   int status = PV_SUCCESS;

   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, initObj, "input/CopyConnInitializeTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, initObj, "input/CopyConnInitializeNonsharedTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, initObj, "input/CopyConnPlasticTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, initObj, "input/CopyConnPlasticNonsharedTest.params"); }

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
int runparamsfile(int argc, char ** argv, PV_Init* initObj, char const * paramsfile) {
   //
   // argv should not contain a "-p" argument.  buildandrun is called with the argv arguments augmented by "-p" and paramsfile
   //
   assert(pv_getopt_str(argc, argv, "-p", NULL, NULL)==-1);
   int rank = 0;
#ifdef PV_USE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI
   int pv_argc = 2+argc;
   char ** pv_argv = (char **) malloc((pv_argc+1)*sizeof(char *));
   assert(pv_argv!=NULL);
   int pv_arg=0;
   for (pv_arg = 0; pv_arg < argc; pv_arg++) {
      pv_argv[pv_arg] = strdup(argv[pv_arg]);
      assert(pv_argv[pv_arg]);
   }
   assert(pv_arg==argc);
   pv_argv[pv_arg++] = strdup("-p");
   pv_argv[pv_arg++] = strdup(paramsfile);
   assert(pv_arg==pv_argc && pv_arg==argc+2);
   assert(pv_argv[argc]!=NULL && pv_argv[argc+1]!=NULL);
   pv_argv[pv_argc]=NULL;


   initObj->initialize(pv_argc, pv_argv);

   int status = PV_SUCCESS;
   HyPerCol * hc = build((int) pv_argc, pv_argv, initObj);
   if (hc != NULL) {
      status = hc->run();
      if( status != PV_SUCCESS ) {
         if (rank==0) {
            fprintf(stderr, "%s: running with params file %s returned error %d.\n", pv_argv[0], paramsfile, status);
         }
      }
   }
   else {
      status = PV_FAILURE;
   }

   for (int arg=0; arg<pv_argc; arg++) {
       free(pv_argv[arg]);
   }
   free(pv_argv);

   if (status != PV_SUCCESS) {
      delete hc;
      return status;
   }

   HyPerConn * origConn = dynamic_cast<HyPerConn *>(hc->getConnFromName("OriginalConn"));
   if (origConn==NULL) {
      if (rank==0) {
         fprintf(stderr, "Unable to find connection named \"OriginalConn\" in params file \"%s\".\n", paramsfile);
      }
      status = PV_FAILURE;
   }
   CopyConn * copyConn = dynamic_cast<CopyConn *>(hc->getConnFromName("CopyConn"));
   if (origConn==NULL) {
      if (rank==0) {
         fprintf(stderr, "Unable to find connection named \"CopyConn\" in params file \"%s\".\n", paramsfile);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      delete hc;
      return status;
   }

   NormalizeBase * origNormalizer = origConn->getNormalizer();
   assert(origNormalizer);
   float origStrength = origNormalizer->getStrength();

   NormalizeBase * copyNormalizer = copyConn->getNormalizer();
   assert(copyNormalizer);
   float copyStrength = copyNormalizer->getStrength();

   int origNumPatches = origConn->getNumDataPatches();
   int copyNumPatches = copyConn->getNumDataPatches();
   assert(origNumPatches==copyNumPatches);
   int origNxp = origConn->xPatchSize();
   int copyNxp = copyConn->xPatchSize();
   assert(origNxp==copyNxp);
   int origNyp = origConn->yPatchSize();
   int copyNyp = copyConn->yPatchSize();
   assert(origNyp==copyNyp);
   int origNfp = origConn->fPatchSize();
   int copyNfp = copyConn->fPatchSize();
   assert(origNfp==copyNfp);
   int origNumArbors = origConn->numberOfAxonalArborLists();
   int copyNumArbors = copyConn->numberOfAxonalArborLists();
   assert(origNumArbors==copyNumArbors);

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
                     fprintf(stderr, "Rank %d: arbor %d, patchindex %d, x=%d, y=%d, f=%d: discrepancy of %g\n",
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
