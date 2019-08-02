#include "checkpointing/CheckpointEntry.hpp"
#include "checkpointing/Checkpointer.hpp"
#include "columns/CommandLineArguments.hpp"
#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "utils/PVLog.hpp"
#include <cerrno>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

int main(int argc, char *argv[]) {
   // Initialize PetaVision environment
   PV::CommandLineArguments arguments{argc, argv, false /*do not allow unrecognized arguments*/};
   MPI_Init(&argc, &argv);
   PV::Communicator const *comm = new PV::Communicator(&arguments);
   PV::MPIBlock const *mpiBlock = comm->getLocalMPIBlock();

   // Params file
   PV::PVParams *params = new PV::PVParams("input/DeleteOldCheckpointsTest.params", 1, comm);

   // Create checkpointing directory and delete any existing files inside it.
   char const *checkpointWriteDir = params->stringValue("checkpointer", "checkpointWriteDir");
   FatalIf(
         checkpointWriteDir == nullptr,
         "Group \"checkpointer\" must have a checkpointWriteDir string parameter.\n");
   std::string checkpointWriteDirectory(checkpointWriteDir);
   ensureDirExists(mpiBlock, checkpointWriteDirectory.c_str());
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
   FatalIf(
         params->value("checkpointer", "deleteOlderCheckpoints") == 0.0,
         "Params file must set deleteOlderCheckpoints to true.\n");
   std::size_t const numKept = (std::size_t)params->valueInt("checkpointer", "numCheckpointsKept");

   // Initialize Checkpointer object
   PV::Checkpointer *checkpointer = new PV::Checkpointer("checkpointer", mpiBlock, &arguments);
   checkpointer->ioParams(PV::PARAMS_IO_READ, params);
   delete params;

   int status = PV_SUCCESS;
   std::vector<std::string> checkpointsCreated;
   for (double t = 0; t < 10; t++) {
      checkpointer->checkpointWrite(t);
      if (mpiBlock->getRank() == 0) {
         auto iter = checkpointsCreated.emplace(
               checkpointsCreated.end(), std::string(checkpointWriteDirectory));
         std::string &newCheckpoint = *iter;
         newCheckpoint.append("/").append("Checkpoint").append("0").append(std::to_string((int)t));
         std::vector<std::string>::iterator firstKept = checkpointsCreated.begin();
         if (checkpointsCreated.size() > numKept) {
            firstKept += (checkpointsCreated.size() - numKept);
         }
         for (std::vector<std::string>::iterator i = checkpointsCreated.begin();
              i != checkpointsCreated.end();
              i++) {
            struct stat dirstat;
            int statResult = stat(i->c_str(), &dirstat);
            if (i < firstKept) {
               if (statResult == 0) {
                  ErrorLog() << (i->c_str()) << " exists but should be deleted.\n";
                  status = PV_FAILURE;
               }
               else {
                  pvAssert(statResult == -1);
                  if (errno != ENOENT) {
                     ErrorLog() << "stat " << (i->c_str()) << " returned \"" << std::strerror(errno)
                                << "\".\n";
                     status = PV_FAILURE;
                  }
               }
            }
            else {
               if (statResult == 0) {
                  bool isDirectory = S_ISDIR(dirstat.st_mode);
                  if (!isDirectory) {
                     ErrorLog() << (i->c_str()) << " exists but is not a directory.\n";
                     status = PV_FAILURE;
                  }
               }
               else {
                  ErrorLog() << "stat " << (i->c_str()) << " returned \"" << std::strerror(errno)
                             << "\".\n";
                  status = PV_FAILURE;
               }
            }
         }
      }
   }
   MPI_Bcast(&status, 1 /*count*/, MPI_INT, 0, mpiBlock->getComm());

   delete checkpointer;
   delete comm;
   MPI_Finalize();

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
