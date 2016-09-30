#include "testDataNoBroadcast.hpp"
#include "testDataWithBroadcast.hpp"
#include "testPvpBatch.hpp"
#include "testPvpExtended.hpp"
#include "testPvpRestricted.hpp"
#include "testSeparatedName.hpp"

#include "columns/Communicator.hpp"
#include "columns/PV_Arguments.hpp"
#include "io/fileio.hpp"
#include "utils/PVLog.hpp"

int main(int argc, char *argv[]) {
   PV::PV_Arguments pvArguments{argc, argv, false /*do not allow unrecognized arguments*/};
   MPI_Init(&argc, &argv);
   PV::Communicator *comm = new PV::Communicator(&pvArguments);

   std::string directory("checkpoints");
   ensureDirExists(comm, directory.c_str()); // Must be called by all processes, because it
                                             // broadcasts the result of the stat() call.
   if (comm->commRank() == 0) {
      std::string rmcommand("rm -rf ");
      rmcommand.append(directory).append("/*");
      pvInfo() << "Cleaning output directory with \"" << rmcommand << "\".\n";
      int rmstatus = system(rmcommand.c_str());
      pvErrorIf(
            rmstatus,
            "Error executing \"%s\": status code was %d\n",
            rmcommand.c_str(),
            WEXITSTATUS(rmstatus));
   }
   if (comm->numCommBatches() > 1) {
      directory.append("/batchsweep_").append(std::to_string(comm->commBatch()));
      ensureDirExists(comm, directory.c_str());
   }

   testSeparatedName(comm);
   testDataNoBroadcast(comm, directory);
   testDataWithBroadcast(comm, directory);
   testPvpRestricted(comm, directory);
   testPvpExtended(comm, directory);
   testPvpBatch(comm, directory);

   delete comm;
   MPI_Finalize();
}
