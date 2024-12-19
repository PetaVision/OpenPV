#include "testDataNoBroadcast.hpp"
#include "testDataWithBroadcast.hpp"
#include "testPvpBatch.hpp"
#include "testPvpBroadcastLayer.hpp"
#include "testPvpExtended.hpp"
#include "testPvpRestricted.hpp"
#include "testSeparatedName.hpp"

#include "columns/Communicator.hpp"
#include "columns/PV_Init.hpp"
#include "io/fileio.hpp"
#include "io/FileManager.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/PVLog.hpp"

int run(PV::PV_Init const &pv_init_obj);

int main(int argc, char *argv[]) {
   PV::PV_Init pv_init_obj(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   int status = run(pv_init_obj);
   return status;
}

int run(PV::PV_Init const &pv_init_obj) {
   PV::Communicator *comm = pv_init_obj.getCommunicator();

   std::string directory("checkpoints");
   auto mpiBlock = comm->getLocalMPIBlock();
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
   }

   auto fileManager = std::make_shared<PV::FileManager>(mpiBlock, directory);
   testSeparatedName(mpiBlock);
   testDataNoBroadcast(fileManager);
   testDataWithBroadcast(fileManager);
   testPvpBroadcastLayer(fileManager);
   testPvpRestricted(fileManager);
   testPvpExtended(fileManager);
   testPvpBatch(fileManager);
   return PV_SUCCESS;
}
