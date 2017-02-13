#include "testDataNoBroadcast.hpp"
#include "testDataWithBroadcast.hpp"
#include "testPvpBatch.hpp"
#include "testPvpExtended.hpp"
#include "testPvpRestricted.hpp"
#include "testSeparatedName.hpp"

#include "columns/CommandLineArguments.hpp"
#include "columns/Communicator.hpp"
#include "io/fileio.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/PVLog.hpp"

int main(int argc, char *argv[]) {
   PV::CommandLineArguments arguments{argc, argv, false /*do not allow unrecognized arguments*/};
   MPI_Init(&argc, &argv);
   PV::Communicator *comm = new PV::Communicator(&arguments);

   std::string directory("checkpoints");
   auto *mpiBlock = comm->getLocalMPIBlock();
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
   if (comm->numCommBatches() > 1) {
      directory.append("/batchsweep_").append(std::to_string(comm->commBatch()));
      ensureDirExists(mpiBlock, directory.c_str());
   }

   testSeparatedName(mpiBlock);
   testDataNoBroadcast(mpiBlock, directory);
   testDataWithBroadcast(mpiBlock, directory);
   testPvpRestricted(mpiBlock, directory);
   testPvpExtended(mpiBlock, directory);
   testPvpBatch(mpiBlock, directory);

   delete comm;
   MPI_Finalize();
}
