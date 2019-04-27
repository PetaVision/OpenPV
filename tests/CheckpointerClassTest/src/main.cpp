#include "checkpointing/CheckpointEntry.hpp"
#include "checkpointing/Checkpointer.hpp"
#include "columns/CommandLineArguments.hpp"
#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "utils/PVLog.hpp"
#include <vector>

int main(int argc, char *argv[]) {
   PV::CommandLineArguments arguments{argc, argv, false /*do not allow unrecognized arguments*/};
   MPI_Init(&argc, &argv);
   PV::Communicator const *comm = new PV::Communicator(&arguments);
   PV::MPIBlock const *mpiBlock = comm->getLocalMPIBlock();

   PV::Checkpointer *checkpointer = new PV::Checkpointer("checkpointer", mpiBlock, &arguments);

   PV::PVParams *params = new PV::PVParams("input/CheckpointerClassTest.params", 1, comm);

   char const *checkpointWriteDir = params->stringValue("checkpointer", "checkpointWriteDir");
   FatalIf(
         checkpointWriteDir == nullptr,
         "Group \"checkpointer\" must have a checkpointWriteDir string parameter.\n");
   std::string checkpointWriteDirectory(checkpointWriteDir);
   ensureDirExists(mpiBlock, checkpointWriteDirectory.c_str()); // Must be called by all processes,
   // because it broadcasts the result of
   // the stat() call.
   if (mpiBlock->getRank() == 0) {
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
   checkpointer->ioParams(PV::PARAMS_IO_READ, params);
   delete params;

   std::vector<double> fpCorrect{1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
   int integerCorrect = 7;

   std::vector<double> fpCheckpoint;
   fpCheckpoint.resize(fpCorrect.size());
   int integerCheckpoint = -5;

   auto floatingpointCheckpointEntry = std::make_shared<PV::CheckpointEntryData<double>>(
         std::string("floatingpoint"),
         mpiBlock,
         fpCheckpoint.data(),
         fpCheckpoint.size(),
         true /*broadcasting*/);

   auto integerCheckpointEntry = std::make_shared<PV::CheckpointEntryData<int>>(
         std::string("integer"), mpiBlock, &integerCheckpoint, (size_t)1, true /*broadcasting*/);

   checkpointer->registerCheckpointEntry(
         floatingpointCheckpointEntry, false /*treat as non-constant*/);
   checkpointer->registerCheckpointEntry(integerCheckpointEntry, false /*treat as non-constant*/);

   checkpointer->checkpointWrite(0.0);
   checkpointer->checkpointWrite(1.0);
   checkpointer->checkpointWrite(3.0);
   checkpointer->checkpointWrite(6.0);

   for (std::vector<double>::size_type n = 0; n < fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = fpCorrect.at(n);
   }
   integerCheckpoint = integerCorrect;
   checkpointer->checkpointWrite(10.0);

   // Clobber the values so that we know checkpointRead is doing something.
   double readTime   = -1.0;
   long int readStep = -1L;
   for (std::vector<double>::size_type n = 0; n < fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = -99.95;
   }
   integerCheckpoint = -25;
   delete checkpointer;
   checkpointer = nullptr;

   // Create a new checkpointer for reading from the checkpoint.
   std::string checkpointReadDir(checkpointWriteDirectory);
   checkpointReadDir.append("/Checkpoint04");
   arguments.setStringArgument("CheckpointReadDirectory", checkpointReadDir);
   checkpointer = new PV::Checkpointer("checkpointer", mpiBlock, &arguments);
   checkpointer->registerCheckpointEntry(
         floatingpointCheckpointEntry, false /*treat as non-constant*/);
   checkpointer->registerCheckpointEntry(integerCheckpointEntry, false /*treat as non-constant*/);

   // Read the values in from checkpoint.
   checkpointer->checkpointRead(&readTime, &readStep);

   // Verify that the values read in are correct.
   int status = PV_SUCCESS;
   if (integerCheckpoint != integerCorrect) {
      ErrorLog().printf(
            "Rank %d, integer checkpoint. Correct value is %d; value read was %d\n",
            mpiBlock->getRank(),
            integerCorrect,
            integerCheckpoint);
      status = PV_FAILURE;
   }

   for (size_t n = 0; n < fpCorrect.size(); n++) {
      if (fpCheckpoint[n] != fpCorrect[n]) {
         ErrorLog().printf(
               "Rank %d, floating-point checkpoint, index %zu. Correct value is %f; value read was "
               "%f\n",
               mpiBlock->getRank(),
               n,
               fpCorrect[n],
               fpCheckpoint[n]);
         status = PV_FAILURE;
      }
   }

   delete checkpointer;
   delete comm;
   MPI_Finalize();

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
