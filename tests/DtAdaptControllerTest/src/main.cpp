/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>

int assertAllZeroes(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   PV_Init initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   int rank = initObj.getWorldRank();

   char const * paramFile1 = "input/DtAdaptController_ANNNormalized.params";
   char const * paramFile2 = "input/DtAdaptController_dtAdaptController.params";
   char const * paramFileCompare = "input/DtAdaptController_Comparison.params";
   int status = PV_SUCCESS;
   if (pv_getopt_str(argc, argv, "-p", NULL, NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument, as it uses hard-coded params files.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   PV_Arguments * arguments = initObj.getArguments();

   arguments->setParamsFile(paramFile1);
   status = rebuildandrun(&initObj, NULL, NULL, NULL, 0);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile1, status);
      exit(status);
   }

   arguments->setParamsFile(paramFile2);
   status = rebuildandrun(&initObj, NULL, NULL, NULL, 0);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFile2, status);
      exit(status);
   }

   arguments->setParamsFile(paramFileCompare);
   status = rebuildandrun(&initObj, NULL, &assertAllZeroes, NULL, 0);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: rank %d running with params file %s returned error %d.\n", arguments->getProgramName(), rank, paramFileCompare, status);
      exit(status);
   }
   
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int assertAllZeroes(HyPerCol * hc, int argc, char * argv[]) {
   const char * targetLayerName = "Comparison";
   HyPerLayer * layer = hc->getLayerFromName(targetLayerName);
   assert(layer);
   LayerProbe * probe = NULL;
   int np = layer->getNumProbes();
   for (int p=0; p<np; p++) {
      if (!strcmp(layer->getProbe(p)->getName(), "ComparisonTest")) {
         probe = layer->getProbe(p);
         break;
      }
   }
   RequireAllZeroActivityProbe * allzeroProbe = dynamic_cast<RequireAllZeroActivityProbe *>(probe);
   assert(allzeroProbe);
   if (allzeroProbe->getNonzeroFound()) {
      if (hc->columnId()==0) {
         double t = allzeroProbe->getNonzeroTime();
         fprintf(stderr, "%s \"%s\" had at least one nonzero activity value, beginning at time %f\n",
               layer->getKeyword(), targetLayerName, t);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}
