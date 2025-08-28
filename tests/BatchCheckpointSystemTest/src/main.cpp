/*
 * main .cpp file for CheckpointSystemTest
 *
 */

#include "CPTestInputLayer.hpp"
#include "VaryingHyPerConn.hpp"
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>

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

   std::string customDefaultsPath("input/DefaultParams.txt");
   initObj.setParams(paramFile1);
   initObj.registerDefaults(customDefaultsPath);
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
   initObj.registerDefaults(customDefaultsPath);
   status = rebuildandrun(&initObj);
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
   auto arguments = hc->getPV_InitObj()->getArguments();
   int rank       = hc->getCommunicator()->getIOMPIBlock()->getRank();
   int rootproc   = 0;

   int status = PV_SUCCESS;
   if (rank == rootproc) {
      long index = hc->getFinalStep();
      std::string cpdir1("checkpoints1");
      PVParams *params = hc->getPV_InitObj()->getParams();
      auto paramsIO = params->makeParamsIO("column");
      std::string cpdir2 = paramsIO->readValue<std::string>("checkpointWriteDir");

      std::string shellcommand("diff -r -q -x timers.txt -x pv?.params -x pv?.params.lua ");
      shellcommand.append(cpdir1).append("/Checkpoint").append(std::to_string(index)).append(" ");
      shellcommand.append(cpdir2).append("/Checkpoint").append(std::to_string(index));
      status = system(shellcommand.c_str());
      if (status != 0) {
         ErrorLog().printf(
               "system(\"%s\") returned %d\n", shellcommand.c_str(), WEXITSTATUS(status));
         status = PV_FAILURE;
      }
   }
   MPI_Bcast(&status, 1, MPI_INT, rootproc, hc->getCommunicator()->communicator());
   return status;
}
