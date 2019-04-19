/*
 * main .cpp file for CheckpointSystemTest
 *
 */

#include "CPTestInputLayer.hpp"
#include "VaryingHyPerConn.hpp"
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

int customexit(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char *argv[]) {
   int rank = 0;
   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char const *paramFile1 = "input/CheckpointParameters1.params";
   char const *paramFile2 = "input/CheckpointParameters2.params";
   int status             = PV_SUCCESS;
   if (!initObj.getStringArgument("ParamsFile").empty()) {
      if (rank == 0) {
         ErrorLog().printf("%s should be run without the params file argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (!initObj.getStringArgument("CheckpointReadDirectory").empty()) {
      if (rank == 0) {
         ErrorLog().printf(
               "%s should be run without the checkpoint directory argument.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (initObj.getBooleanArgument("Restart")) {
      if (rank == 0) {
         ErrorLog().printf("%s should be run without the restart flag.\n", argv[0]);
      }
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      if (rank == 0) {
         ErrorLog().printf(
               "This test uses two hard-coded params files, %s and %s. The second run is started "
               "from a checkpoint from the first run, and the results of the two runs are "
               "compared.\n",
               paramFile1,
               paramFile2);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   if (rank == 0) {
      char const *rmcommand = "rm -rf checkpoints1 checkpoints2 output";
      status                = system(rmcommand);
      if (status != 0) {
         Fatal().printf(
               "deleting old checkpoints and output directories failed: \"%s\" returned %d\n",
               rmcommand,
               status);
      }
   }

   initObj.registerKeyword("CPTestInputLayer", Factory::create<CPTestInputLayer>);
   initObj.registerKeyword("VaryingHyPerConn", Factory::create<VaryingHyPerConn>);

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
   initObj.setStringArgument("CheckpointReadDirectory", "checkpoints1/Checkpoint12");

   status = rebuildandrun(&initObj, nullptr, customexit);
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "%s: rank %d running with params file %s returned error %d.\n",
            initObj.getProgramName(),
            rank,
            paramFile2,
            status);
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char *argv[]) {
   // Rank of the checkpointing MPI communicator does is not publicly accessible, so recreate it.
   Arguments const *arguments = hc->getPV_InitObj()->getArguments();
   MPIBlock mpiBlock(
         hc->getCommunicator()->globalCommunicator(),
         arguments->getIntegerArgument("NumRows"),
         arguments->getIntegerArgument("NumColumns"),
         arguments->getIntegerArgument("BatchWidth"),
         arguments->getIntegerArgument("CheckpointCellNumRows"),
         arguments->getIntegerArgument("CheckpointCellNumColumns"),
         arguments->getIntegerArgument("CheckpointCellBatchDimension"));
   int rank     = mpiBlock.getRank();
   int rootproc = 0;

   int status = PV_SUCCESS;
   if (rank == rootproc) {
      long index            = hc->getFinalStep();
      const char *cpdir1    = "checkpoints1";
      const char *cpdir2    = hc->parameters()->stringValue("column", "checkpointWriteDir");
      const int max_buf_len = 1024;
      char shellcommand[max_buf_len];
      const char *fmtstr = "diff -r -q -x timers.txt -x pv.params -x pv.params.lua "
                           "%s/Checkpoint%ld %s/Checkpoint%ld";
      snprintf(shellcommand, max_buf_len, fmtstr, cpdir1, index, cpdir2, index);
      status = system(shellcommand);
      if (status != 0) {
         ErrorLog().printf("system(\"%s\") returned %d\n", shellcommand, WEXITSTATUS(status));
         status = PV_FAILURE;
      }
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->getCommunicator()->communicator());
   return status;
}
