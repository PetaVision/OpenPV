/*
 * LayerRestartTest.cpp
 *
 * Tests the restart flag for HyPerLayer.
 * Run without arguments, it will run the following params files in sequence:
 * input/LayerRestartTest-Write.params
 * input/LayerRestartTest-Check.params.
 * input/LayerRestartTest-Read.params.
 *
 * LayerRestartTest-Write loads an image from input/F_N160050.jpg into
 * the layer "Copy" and random noise into the layer "Comparison", and
 * writes Copy_{A,V}.pvp and Comparison_{A,V}.pvp files.
 *
 * LayerRestartTest-Check restarts the Comparison layer.
 * the main() function calls buildandrun with this params file,
 * and uses the customexit hook to verify that Comparison is
 * not all zeros.
 *
 * LayerRestartTest-Read restarts the Copy and Comparison layers,
 * and loads input/F_N160050.jpg into an Image layer.
 * The comparison layer takes the difference of Image and Copy,
 * which should be all zeros.  The customexit hook now verifies
 * that Comparison is all zeros.
 *
 */

#include "arch/mpi/mpi.h"
#include "columns/buildandrun.hpp"
#include "layers/HyPerLayer.hpp"

int checkComparisonZero(HyPerCol *hc, int argc, char *argv[]);
int checkComparisonNonzero(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int status;
   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (initObj.getParams() != nullptr) {
      if (rank == 0) {
         ErrorLog().printf(
               "%s runs a number of params files in sequence.  Do not include a '-p' option when "
               "running this program.\n",
               argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD); /* Can't use `initObj.getComm()->communicator()` because
                                      initObj.initialize hasn't been called. */
      exit(EXIT_FAILURE);
   }

   initObj.setParams("input/LayerRestartTest-Write.params");
   status = rebuildandrun(&initObj);
   if (status == PV_SUCCESS) {
      char const *checkParamsFile = "input/LayerRestartTest-Check.params";
      if (rank == 0) {
         InfoLog().printf(
               "*** %s: running params file %s\n", initObj.getProgramName(), checkParamsFile);
      }
      initObj.setParams("input/LayerRestartTest-Check.params");
      status = rebuildandrun(&initObj, NULL, &checkComparisonNonzero);
      if (status == PV_SUCCESS) {
         char const *readParamsFile = "input/LayerRestartTest-Read.params";
         if (rank == 0) {
            InfoLog().printf(
                  "*** %s: running params file %s\n", initObj.getProgramName(), checkParamsFile);
         }
         initObj.setParams(readParamsFile);
         status = rebuildandrun(&initObj, NULL, &checkComparisonZero);
      }
   }

#ifdef PV_USE_MPI
   // Output status from each process, but go through root process since we might be using MPI
   // across several machines
   // and only have a console on the root process
   if (rank == 0) {
      int otherprocstatus = status;
      int commsize;
      MPI_Comm_size(MPI_COMM_WORLD, &commsize);
      for (int r = 0; r < commsize; r++) {
         if (r != 0)
            MPI_Recv(&otherprocstatus, 1, MPI_INT, r, 59, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         if (otherprocstatus == PV_SUCCESS) {
            InfoLog().printf("%s: rank %d process succeeded.\n", argv[0], r);
         }
         else {
            ErrorLog().printf(
                  "%s: rank %d process FAILED with return code %d\n", argv[0], r, otherprocstatus);
            status = PV_FAILURE;
         }
      }
   }
   else {
      MPI_Send(&status, 1, MPI_INT, 0, 59, MPI_COMM_WORLD);
   }

// if( !mpi_initialized_on_entry ) MPI_Finalize();
#endif // PV_USE_MPI
   return status;
}

int checkComparisonNonzero(HyPerCol *hc, int argc, char *argv[]) {
   int status        = PV_FAILURE;
   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("Comparison"));
   FatalIf(layer == nullptr, "No layer \"Comparison\".\n");
   float *V = layer->getV();
   for (int k = 0; k < layer->getNumNeurons(); k++) {
      if (V[k]) {
         status = PV_SUCCESS;
         break;
      }
   }
   return status;
}

int checkComparisonZero(HyPerCol *hc, int argc, char *argv[]) {
   int status        = PV_SUCCESS;
   HyPerLayer *layer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("Comparison"));
   FatalIf(layer == nullptr, "No layer \"Comparison\".\n");
   float *V = layer->getV();
   for (int k = 0; k < layer->getNumNeurons(); k++) {
      if (V[k]) {
         ErrorLog().printf("Neuron %d: discrepancy %f\n", k, (double)V[k]);
         status = PV_FAILURE;
      }
   }
   return status;
}
