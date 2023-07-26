/*
 * DryRunFlagTest.cpp
 *
 */

#include "AlwaysFailsLayer.hpp"
#include <columns/buildandrun.hpp>
#include <columns/Factory.hpp>
#include <sys/types.h>
#include <unistd.h>
#include <utils/CompareParamsFiles.hpp>

int deleteOutputDirectory(PV::Communicator const *comm);

int main(int argc, char *argv[]) {

   int status = PV_SUCCESS;

   PV::PV_Init pv_obj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   pv_obj.registerKeyword("AlwaysFailsLayer", Factory::create<AlwaysFailsLayer>);

   pv_obj.setBooleanArgument("DryRun", true);

   if (pv_obj.isExtraProc()) {
      return EXIT_SUCCESS;
   }

   FatalIf(
         pv_obj.getParams() != nullptr,
         "%s should be called without the -p argument; the necessary params file is hard-coded.\n",
         argv[0]);
   pv_obj.setParams("input/DryRunFlagTest.params");

   auto *comm = pv_obj.getCommunicator();

   status = deleteOutputDirectory(comm);
   if (status != PV_SUCCESS) {
      Fatal().printf("%s: error cleaning generated files from any previous run.\n", argv[0]);
   }

   status = buildandrun(&pv_obj);

   if (status != PV_SUCCESS) {
      int rank = comm->globalCommRank();
      Fatal().printf("%s: running with dry-run flag set failed on process %d.\n", argv[0], rank);
   }

   status = PV::compareParamsFiles(
         std::string("output/pv.params"),
         std::string("input/correct.params"),
         comm->globalCommunicator());
   if (status != PV_SUCCESS) {
      Fatal().printf("%s failed.\n", argv[0]);
   }
   return status;
}

int deleteOutputDirectory(PV::Communicator const *comm) {
   int status = PV_SUCCESS;
   if (comm->globalCommRank() == 0) {
      if (system("rm -rf output") != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   MPI_Bcast(&status, 1, MPI_INT, 0, comm->communicator());
   return status;
}
