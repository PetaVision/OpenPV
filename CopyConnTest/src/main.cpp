/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <connections/CopyConn.hpp>
#include <connections/HyPerConn.hpp>
#include <normalizers/NormalizeBase.hpp>

int runparamsfile(int argc, char ** argv, char const * paramsfile);

int main(int argc, char * argv[]) {
   int rank = 0;
#ifdef PV_USE_MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI
   int status = PV_SUCCESS;

   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, "input/CopyConnInitializeTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, "input/CopyConnInitializeNonsharedTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, "input/CopyConnPlasticTest.params"); }
   if (status == PV_SUCCESS) { status = runparamsfile(argc, argv, "input/CopyConnPlasticNonsharedTest.params"); }

#ifdef PV_USE_MPI
   MPI_Finalize();
#endif // PV_USE_MPI
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
int runparamsfile(int argc, char ** argv, char const * paramsfile) {
   //
   // Process input arguments.  The only input arguments allowed are --require-return and -t.
   // The parameter file is hardcoded below.
   //
   int rank = 0;
#ifdef PV_USE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI
   bool argerr = false;
   int reqrtn = 0;
   int usethreads = 0;
   int threadargno = -1;
   for (int k=1; k<argc; k++) {
      if (!strcmp(argv[k], "--require-return")) {
         reqrtn = 1;
      }
      else if (!strcmp(argv[k], "-t")) {
         usethreads = 1;
         if (k<argc-1 && argv[k+1][0] != '-') {
            k++;
            threadargno = k;
         }
      }
      else {
         argerr = true;
         break;
      }
   }
   if (argerr) {
      if (rank==0) {
         fprintf(stderr, "%s: run without input arguments (except for --require-return, if desired); the necessary arguments are hardcoded.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      return PV_FAILURE;
   }

   assert(reqrtn==0 || reqrtn==1);
   assert(usethreads==0 || usethreads==1);
   size_t cl_argc = 3+reqrtn+usethreads+(threadargno>0);
   char ** cl_args = (char **) malloc((cl_argc+1)*sizeof(char *));
   assert(cl_args!=NULL);
   int cl_arg = 0;
   cl_args[cl_arg++] = strdup(argv[0]);
   cl_args[cl_arg++] = strdup("-p");
   cl_args[cl_arg++] = strdup(paramsfile);
   if (reqrtn) {
      cl_args[cl_arg++] = strdup("--require-return");
   }
   if (usethreads) {
      cl_args[cl_arg++] = strdup("-t");
      if (threadargno>0) {
         cl_args[cl_arg++] = strdup(argv[threadargno]);
      }
   }
   assert(cl_arg==cl_argc);
   cl_args[cl_arg] = NULL;
   int status = PV_SUCCESS;
   HyPerCol * hc = build((int) cl_argc, cl_args);
   if (hc != NULL) {
      status = hc->run();
      if( status != PV_SUCCESS ) {
         if (rank==0) {
            fprintf(stderr, "%s: running with params file %s returned error %d.\n", cl_args[0], paramsfile, status);
         }
      }
   }
   else {
      status = PV_FAILURE;
   }

   for (size_t arg=0; arg<cl_argc; arg++) {
       free(cl_args[arg]);
   }
   free(cl_args);

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
