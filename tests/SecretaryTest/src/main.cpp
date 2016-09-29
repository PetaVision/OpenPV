#include "columns/Communicator.hpp"
#include "columns/PV_Arguments.hpp"
#include "io/CheckpointEntry.hpp"
#include "io/io.hpp"
#include "io/PVParams.hpp"
#include "io/Secretary.hpp"
#include "utils/PVLog.hpp"
#include <vector>

int main(int argc, char * argv[]) {
   PV::PV_Arguments pvArguments{argc, argv, false/*do not allow unrecognized arguments*/};
   MPI_Init(&argc, &argv);
   PV::Communicator * comm = new PV::Communicator(&pvArguments);

   PV::Secretary * secretary = new PV::Secretary("secretary", comm);

   PV::PVParams * params = new PV::PVParams("input/SecretaryTest.params", 1, comm);

   char const * checkpointWriteDir = params->stringValue("secretary", "checkpointWriteDir");
   pvErrorIf(checkpointWriteDir==nullptr, "Group \"secretary\" must have a checkpointWriteDir string parameter.\n");
   std::string checkpointWriteDirectory(checkpointWriteDir);
   ensureDirExists(comm, checkpointWriteDirectory.c_str()); // Must be called by all processes, because it broadcasts the result of the stat() call.
   if (comm->commRank()==0) {
      std::string rmcommand("rm -rf "); rmcommand.append(checkpointWriteDirectory).append("/*");
      pvInfo() << "Cleaning directory \"" << checkpointWriteDirectory << "\" with \"" << rmcommand << "\".\n";
      int rmstatus = system(rmcommand.c_str());
      pvErrorIf(rmstatus, "Error executing \"%s\": status code was %d\n", rmcommand.c_str(), WEXITSTATUS(rmstatus));
   }
   secretary->ioParamsFillGroup(PV::PARAMS_IO_READ, params);
   delete params;

   std::vector<double> fpCorrect{1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
   int integerCorrect = 7;

   std::vector<double> fpCheckpoint; fpCheckpoint.resize(fpCorrect.size());
   int integerCheckpoint = -5;

   secretary->registerCheckpointEntry(std::make_shared<PV::CheckpointEntryData<double> >(
         std::string("floatingpoint"), comm, fpCheckpoint.data(), fpCheckpoint.size(), true/*broadcasting*/));
   secretary->registerCheckpointEntry(std::make_shared<PV::CheckpointEntryData<int> >(
         std::string("integer"), comm, &integerCheckpoint, (size_t) 1, true/*broadcasting*/));

   secretary->checkpointWrite(0.0);
   secretary->checkpointWrite(1.0);
   secretary->checkpointWrite(3.0);
   secretary->checkpointWrite(6.0);

   for (std::vector<double>::size_type n=0; n<fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = fpCorrect.at(n);
   }
   integerCheckpoint = integerCorrect;
   secretary->checkpointWrite(10.0);

   double readTime = -1.0;
   long int readStep = -1L;
   for (std::vector<double>::size_type n=0; n<fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = -99.95;
   }
   integerCheckpoint = -25;

   std::string checkpointReadDir(checkpointWriteDirectory);
   checkpointReadDir.append("/Checkpoint4");
   secretary->checkpointRead(checkpointReadDir, &readTime, &readStep);

   delete secretary;
   delete comm;
   MPI_Finalize();

   return 0;
}
