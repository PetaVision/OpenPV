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
   secretary->ioParamsFillGroup(PV::PARAMS_IO_READ, params);
   delete params;
   bool verifyWritesFlag = secretary->doesVerifyWrites();

   std::vector<double> fpCorrect{1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
   int integerCorrect = 7;

   std::vector<double> fpCheckpoint; fpCheckpoint.resize(fpCorrect.size());
   int integerCheckpoint = -5;

   secretary->registerCheckpointEntry(std::make_shared<PV::CheckpointEntryData<double> >(
         std::string("floatingpoint"), comm, fpCheckpoint.data(), fpCheckpoint.size(), true/*broadcasting*/));
   secretary->registerCheckpointEntry(std::make_shared<PV::CheckpointEntryData<int> >(
         std::string("integer"), comm, &integerCheckpoint, (size_t) 1, true/*broadcasting*/));

   secretary->checkpointWrite("checkpoint0", 0.0);
   secretary->checkpointWrite("checkpoint1", 1.0);
   secretary->checkpointWrite("checkpoint2", 3.0);
   secretary->checkpointWrite("checkpoint3", 6.0);

   for (std::vector<double>::size_type n=0; n<fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = fpCorrect.at(n);
   }
   integerCheckpoint = integerCorrect;
   secretary->checkpointWrite("checkpoint4", 10.0);

   double readTime = -1.0;
   long int readStep = -1L;
   for (std::vector<double>::size_type n=0; n<fpCheckpoint.size(); n++) {
      fpCheckpoint.at(n) = -99.95;
   }
   integerCheckpoint = -25;

   secretary->checkpointRead("checkpoint4", &readTime, &readStep);

   delete secretary;
   delete comm;
   MPI_Finalize();

   return 0;
}
