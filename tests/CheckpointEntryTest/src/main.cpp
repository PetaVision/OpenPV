#include "testDataNoBroadcast.hpp"
#include "testDataWithBroadcast.hpp"
#include "testPvpBatch.hpp"
#include "testPvpExtended.hpp"
#include "testPvpRestricted.hpp"
#include "testSeparatedName.hpp"

#include "columns/CommandLineArguments.hpp"
#include "columns/Communicator.hpp"
#include "io/fileio.hpp"
#include "io/FileManager.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/PVLog.hpp"

int run(PV::Arguments const &arguments);

int main(int argc, char *argv[]) {
   MPI_Init(&argc, &argv);
   PV::CommandLineArguments arguments{argc, argv, false /*do not allow unrecognized arguments*/};
   int status = run(arguments);
   MPI_Finalize();
   return status;
}

int run(PV::Arguments const &arguments) {
   PV::Communicator const comm = PV::Communicator(&arguments);

   std::string directory("checkpoints");
   auto mpiBlock = comm.getLocalMPIBlock();
   ensureDirExists(mpiBlock, directory.c_str());
   if (mpiBlock->getRank() == 0) {
      std::string rmcommand("rm -rf " + directory + "/*");
      InfoLog() << "Cleaning output directory with \"" << rmcommand << "\".\n";
      int rmstatus = system(rmcommand.c_str());
      FatalIf(
            rmstatus,
            "Error executing \"%s\": status code was %d\n",
            rmcommand.c_str(),
            WEXITSTATUS(rmstatus));
   }
   if (comm.numCommBatches() > 1) {
      directory.append("/batchsweep_").append(std::to_string(comm.commBatch()));
   }

   auto fileManager = std::make_shared<PV::FileManager>(mpiBlock, directory);
   testSeparatedName(mpiBlock);
   testDataNoBroadcast(fileManager);
   testDataWithBroadcast(fileManager);
   testPvpRestricted(fileManager);
   testPvpExtended(fileManager);
   testPvpBatch(fileManager);
   return PV_SUCCESS;
}
