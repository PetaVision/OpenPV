/*
 * main .cpp file for WriteActivitySparseTest
 *
 */
#include "TestNotAlwaysAllZerosProbe.hpp"
#include <arch/mpi/mpi.h>
#include <columns/Factory.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <include/pv_common.h>
#include <layers/HyPerLayer.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>

int checkProbesOnExit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int rank = 0;
   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char const *paramFile1 = "input/GenerateOutput.params";
   char const *paramFile2 = "input/TestOutput.params";
   int status             = PV_SUCCESS;
   if (initObj.getParams() != NULL) {
      if (rank == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s should be run without the params file argument.\n", initObj.getProgramName());
         errorMessage.printf(
               "This test uses two hard-coded params files, %s and %s. The first generates an "
               "output pvp file, and the second checks whether the output is consistent with the "
               "input.\n",
               paramFile1,
               paramFile2);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   if (rank == 0) {
      char const *rmcommand = "rm -rf outputGenerate outputTest";
      status                = system(rmcommand);
      if (status != 0) {
         Fatal().printf(
               "deleting old output directories failed: \"%s\" returned %d\n", rmcommand, status);
      }
   }

   initObj.registerKeyword(
         "TestNotAlwaysAllZerosProbe", Factory::create<TestNotAlwaysAllZerosProbe>);

   initObj.setParams(paramFile1);
   status = rebuildandrun(&initObj);
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "%s: rank %d running with params file %s returned error %d.\n",
            initObj.getProgramName(),
            rank,
            paramFile1,
            status);
   }

   initObj.setParams(paramFile2);

   status = rebuildandrun(&initObj, NULL, &checkProbesOnExit);
   if (status != PV_SUCCESS) {
      ErrorLog().printf(
            "%s: rank %d running with params file %s returned status %d.\n",
            initObj.getProgramName(),
            rank,
            paramFile2,
            status);
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

// check that the input layer has become nonzero at some point
// If the comparison layer ever has a nonzero value, RequireAllZeroActivityProbe
// should catch it and exit with an error.
// However, there are some bugs that could cause InputLayer to
// have all zeros in the activity.  In that case, OriginalInputLayer and
// GeneratedInputLayer would always be zero, so the comparison would always
// be zero, so the test would pass even though there's a bug.
//
// The problem of what to do if comparison reports zero when given
// nonzero input is best left for a different test.
int checkProbesOnExit(HyPerCol *hc, int argc, char *argv[]) {
   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("OriginalMovie"));
   FatalIf(!layer, "No layer named \"OriginalMovie\".\n");
   TestNotAlwaysAllZerosProbe *testNonzero =
         dynamic_cast<TestNotAlwaysAllZerosProbe *>(hc->getObjectFromName("OriginalMovieProbe"));
   FatalIf(!testNonzero, "No probe named \"OriginalMovieProbe\".\n");
   FatalIf(
         !testNonzero->nonzeroValueHasOccurred(),
         "OriginalMovieProbe is always zero.\n",
         layer->getDescription_c());

   return PV_SUCCESS;
}
