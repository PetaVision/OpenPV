#include "checkpointing/CheckpointEntry.hpp"
#include "checkpointing/Checkpointer.hpp"
#include "columns/CommandLineArguments.hpp"
#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "io/io.hpp"
#include "utils/PVLog.hpp"
#include <vector>

int main(int argc, char *argv[]) {
   PV::CommandLineArguments arguments{argc, argv, false /*do not allow unrecognized arguments*/};
   MPI_Init(&argc, &argv);
   PV::Communicator *comm = new PV::Communicator(&arguments);

   PV::Checkpointer *checkpointer = new PV::Checkpointer("checkpointer", comm);

   PV::PVParams *params = new PV::PVParams("input/CheckpointerClassTest.params", 1, comm);

   char const *checkpointWriteDir = params->stringValue("checkpointer", "checkpointWriteDir");
   FatalIf(
         checkpointWriteDir == nullptr,
         "Group \"checkpointer\" must have a checkpointWriteDir string parameter.\n");
   std::string checkpointWriteDirectory(checkpointWriteDir);
   ensureDirExists(comm, checkpointWriteDirectory.c_str()); // Must be called by all processes,
   // because it broadcasts the result of
   // the stat() call.
   if (comm->commRank() == 0) {
      std::string rmcommand("rm -rf ");
      rmcommand.append(checkpointWriteDirectory).append("/*");
      InfoLog() << "Cleaning directory \"" << checkpointWriteDirectory << "\" with \"" << rmcommand
                << "\".\n";
      int rmstatus = system(rmcommand.c_str());
      FatalIf(
            rmstatus,
            "Error executing \"%s\": status code was %d\n",
            rmcommand.c_str(),
            WEXITSTATUS(rmstatus));
   }
   if (comm->numCommBatches() > 1) {
      params->setBatchSweepValues();
      checkpointWriteDirectory =
            std::string(params->stringValue("checkpointer", "checkpointWriteDir"));
   }
   checkpointer->ioParamsFillGroup(PV::PARAMS_IO_READ, params);
   delete params;

   std::vector<double> fpCorrect{1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
   int integerCorrect = 7;

   std::vector<double> fpCheckpoint;
   fpCheckpoint.resize(fpCorrect.size());
   int integerCheckpoint = -5;

   checkpointer->registerCheckpointEntry(
         std::make_shared<PV::CheckpointEntryData<double>>(
               std::string("floatingpoint"),
               comm,
               fpCheckpoint.data(),
               fpCheckpoint.size(),
               true /*broadcasting*/));
   checkpointer->registerCheckpointEntry(
         std::make_shared<PV::CheckpointEntryData<int>>(
               std::string("integer"), comm, &integerCheckpoint, (size_t)1, true /*broadcasting*/));

   checkpointer->checkpointWrite(0.0);
   checkpointer->checkpointWrite(1.0);
   checkpointer->checkpointWrite(3.0);
   checkpointer->checkpointWrite(6.0);

   for (std::vector<double>::size_type n = 0; n < fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = fpCorrect.at(n);
   }
   integerCheckpoint = integerCorrect;
   checkpointer->checkpointWrite(10.0);

   double readTime   = -1.0;
   long int readStep = -1L;
   for (std::vector<double>::size_type n = 0; n < fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = -99.95;
   }
   integerCheckpoint = -25;

   std::string checkpointReadDir(checkpointWriteDirectory);
   checkpointReadDir.append("/Checkpoint04");
   checkpointer->setCheckpointReadDirectory(checkpointReadDir);
   checkpointer->checkpointRead(&readTime, &readStep);

   delete checkpointer;
   delete comm;
   MPI_Finalize();

   return 0;
}
