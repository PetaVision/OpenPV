/*
 * CheckpointerMPIBlockTest.cpp
 *
 */
#include "checkpointing/Checkpointer.hpp"
#include "arch/mpi/mpi.h"
#include "checkpointing/CheckpointEntryPvp.hpp"
#include "columns/ConfigFileArguments.hpp"
#include "include/pv_common.h"
#include "utils/PVLog.hpp"
#include <vector>

int run(int argc, char *argv[]);

int main(int argc, char *argv[]) {
   MPI_Init(nullptr, nullptr);
   int status = run(argc, argv);
   MPI_Finalize();
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run(int argc, char *argv[]) {
   FatalIf(
         argc < 2 || argv == nullptr || argv[1] == nullptr,
         "Config file must be specified on the command line.\n");

   std::string configFile(argv[1]);
   PV::ConfigFileArguments arguments(
         std::string{argv[1]}, MPI_COMM_WORLD, false /*do not allow unrecognized arguments*/);

   PV::Communicator pvComm(&arguments);
   PV::MPIBlock const *globalMPIBlock = pvComm.getGlobalMPIBlock();

   std::string logFile = arguments.getStringArgument("LogFile");
   if (!logFile.empty()) {
      size_t finalSlash = logFile.rfind('/');
      size_t finalDot   = logFile.rfind('.');
      size_t insertionPoint;
      if (finalDot == std::string::npos
          or (finalSlash != std::string::npos and finalDot < finalSlash)) {
         insertionPoint = logFile.length();
      }
      else {
         insertionPoint = finalDot;
      }
      std::string insertion("_");
      insertion.append(std::to_string(globalMPIBlock->getRank()));
      logFile.insert(insertionPoint, insertion);
      PV::setLogFile(logFile);
   }

   int const globalNumRows       = globalMPIBlock->getNumRows();
   int const globalNumColumns    = globalMPIBlock->getNumColumns();
   int const globalMPIBatchWidth = globalMPIBlock->getBatchDimension();

   auto checkpointer =
         new PV::Checkpointer(std::string("checkpointer"), globalMPIBlock, &arguments);

   PV::PVParams params("input/CheckpointerMPIBlockTest.params", 1, &pvComm);
   checkpointer->ioParams(PV::PARAMS_IO_READ, &params);

   // Delete any existing checkpoints directory.
   char const *checkpointDirectory = checkpointer->getCheckpointWriteDir();
   if (checkpointDirectory != nullptr and checkpointDirectory[0] != '\0') {
      for (int r = 0; r < globalMPIBlock->getSize(); r++) {
         if (globalMPIBlock->getRank() == r) {
            struct stat checkpointDirStat;
            int statResult = stat(checkpointDirectory, &checkpointDirStat);
            if (statResult == 0) {
               std::string rmrcommand = std::string("rm -r '") + checkpointDirectory + '\'';
               int systemResult       = system(rmrcommand.c_str());
               FatalIf(
                     systemResult != 0,
                     "Removing previously existing checkpoint directory with \"%s\" failed.\n",
                     rmrcommand);
            }
            else {
               FatalIf(
                     errno != ENOENT,
                     "error getting stat of \"%s\": \"%s\"\n",
                     checkpointDirectory,
                     strerror(errno));
            }
         }
         MPI_Barrier(globalMPIBlock->getComm());
      }
   }

   // Create an 8x8x1 layer with local batch width 3
   // (and overall batch width 3*globalMPIBatchWidth)
   // Data at neuron (m,n) for batch element k will be 100*k + (8*n+m).

   int status = PV_SUCCESS;

   int const layerWidth  = 8;
   int const layerHeight = 8;

   int const nxLocal         = layerWidth / globalNumColumns;
   int const nyLocal         = layerHeight / globalNumRows;
   int const batchWidthLocal = 3;
   if (nxLocal * globalNumColumns != layerWidth) {
      ErrorLog() << "NumColumns in config file must divide " << layerWidth << ".\n";
      status = PV_FAILURE;
   }
   if (nyLocal * globalNumRows != layerHeight) {
      ErrorLog() << "NumRows in config file must divide " << layerHeight << ".\n";
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      return status;
   }

   std::vector<float> correctValues(nxLocal * nyLocal * batchWidthLocal);
   int const columnOffset = globalMPIBlock->getColumnIndex() * nxLocal;
   int const rowOffset    = globalMPIBlock->getRowIndex() * nyLocal;
   int const batchOffset  = globalMPIBlock->getBatchIndex() * batchWidthLocal;
   for (int j = 0; j < (int)correctValues.size(); j++) {
      int const localColumnIndex  = j % nxLocal;
      int const globalColumnIndex = localColumnIndex + columnOffset;

      int const localRowIndex  = (j / nxLocal) % nyLocal; // Integer division
      int const globalRowIndex = localRowIndex + rowOffset;

      int const localBatchIndex  = j / (nxLocal * nyLocal); // Integer devision
      int const globalBatchIndex = localBatchIndex + batchOffset;

      correctValues[j] =
            (float)(100 * globalBatchIndex + layerWidth * globalRowIndex + globalColumnIndex);
   }

   PVLayerLoc loc;
   loc.nbatch       = batchWidthLocal;
   loc.nx           = nxLocal;
   loc.ny           = nyLocal;
   loc.nf           = 1;
   loc.nbatchGlobal = batchWidthLocal * globalMPIBatchWidth;
   loc.nxGlobal     = layerWidth;
   loc.nyGlobal     = layerHeight;
   loc.kb0          = batchOffset;
   loc.kx0          = columnOffset;
   loc.ky0          = rowOffset;
   loc.halo.lt      = 0;
   loc.halo.rt      = 0;
   loc.halo.dn      = 0;
   loc.halo.up      = 0;

   // Copy correctValues to a separate vector, write it to checkpoint, read it back, and compare.
   auto testValues = correctValues;

   auto checkpointEntry = std::make_shared<PV::CheckpointEntryPvp<float>>(
         std::string("TestBuffer"),
         checkpointer->getMPIBlock(),
         testValues.data(),
         &loc,
         false /*not extended*/);

   bool registerSucceeded = checkpointer->registerCheckpointEntry(checkpointEntry);
   FatalIf(!registerSucceeded, "Checkpointer failed to register TestBuffer for checkpointing.\n");

   checkpointer->checkpointWrite(0.0);

   // Clobber testValues, to ensure that checkpointRead is working.
   for (auto &n : testValues) {
      n = -0.5;
   }

   // Create a new checkpointer for reading from the checkpoint.
   std::string checkpointWriteDir = checkpointer->getCheckpointWriteDir();
   std::string checkpointReadDir(checkpointWriteDir);
   checkpointReadDir.append("/Checkpoint0");
   arguments.setStringArgument("CheckpointReadDirectory", checkpointReadDir);

   auto checkpointReader =
         new PV::Checkpointer(std::string("checkpointer"), globalMPIBlock, &arguments);

   registerSucceeded = checkpointReader->registerCheckpointEntry(checkpointEntry);
   FatalIf(!registerSucceeded, "Checkpointer failed to register TestBuffer for checkpointing.\n");

   // Read the values in from checkpoint.
   double timestamp;
   long int stepnumber;
   checkpointReader->checkpointRead(&timestamp, &stepnumber);
   delete checkpointer;
   checkpointer = nullptr;
   // Why does it fail if I delete checkpointer before the checkpointReader->checkpointRead call?
   delete checkpointReader;
   checkpointReader = nullptr;

   // Verify that the values read in are correct.
   FatalIf(
         testValues.size() != correctValues.size(),
         "test vector and correct vector have different sizes.\n");
   for (int j = 0; j < (int)correctValues.size(); j++) {
      if (testValues[j] != correctValues[j]) {
         ErrorLog() << "Global rank " << globalMPIBlock->getRank() << ", local index " << j
                    << ": expected value " << correctValues[j] << "; observed value "
                    << testValues[j] << ".\n";
         status = PV_FAILURE;
      }
   }
   return status;
}
