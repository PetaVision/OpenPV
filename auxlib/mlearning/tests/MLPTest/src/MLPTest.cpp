/*
 *  * main.cpp for MLPTest
 *   *
 *    */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include <columns/MLPRegisterKeywords.hpp>
#include "InputLayer.hpp"
#include "GTLayer.hpp"
#include "ComparisonLayer.hpp"

int main(int argc, char * argv[]) {
   PV_Init initObj(&argc, &argv, false/*allowUnrecognizedArguments*/);
   initObj.initialize();
   PVMLearning::MLPRegisterKeywords(&initObj);
   initObj.registerKeyword("InputLayer", PVMLearning::createInputLayer);
   initObj.registerKeyword("GTLayer", PVMLearning::createGTLayer);
   initObj.registerKeyword("ComparisonLayer", PVMLearning::createComparisonLayer);
   int rank = initObj.getWorldRank();

   if (initObj.getParams() != NULL) {
      if (rank==0) {
         fprintf(stderr, "%s should be run without the params file argument.\n", argv[0]);
         fprintf(stderr, "This test hard-codes the necessary params file.\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   PV_Arguments * arguments = initObj.getArguments();

   int status = PV_SUCCESS;

   const char* paramsFile = "input/GradientCheckW1.params";
   initObj.setParams(paramsFile);
   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/GradientCheckW2.params";
   initObj.setParams(paramsFile);
   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/MLPTrain.params";
   initObj.setParams(paramsFile);
   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/MLPTest.params";
   initObj.setParams(paramsFile);
   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
      exit(status);
   }

   paramsFile = "input/AlexTrain.params";
   initObj.setParams(paramsFile);
   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   paramsFile = "input/AlexTest.params";
   initObj.setParams(paramsFile);
   status = rebuildandrun(&initObj);
   if( status != PV_SUCCESS ) {
      fprintf(stderr, "%s: running with params file %s returned error %d.\n", argv[0], paramsFile, status);
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
