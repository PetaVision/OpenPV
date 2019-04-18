/*
 * TransposeWeightsTest.cpp
 *
 */

#include "TestNonshared.hpp"
#include "TestShared.hpp"
#include <columns/PV_Init.hpp>
#include <utils/TransposeWeights.hpp>

void checkSizeCompatibility(PVLayerLoc const &loc, PV::Communicator *comm);
int testOneToOneShared(PV::Communicator *comm);
int testManyToOneShared(PV::Communicator *comm);
int testOneToManyShared(PV::Communicator *comm);
int testOneToOneNonshared(PV::Communicator *comm);
int testManyToOneNonshared(PV::Communicator *comm);
int testOneToManyNonshared(PV::Communicator *comm);

int main(int argc, char *argv[]) {
   PV::PV_Init *pv_init   = new PV::PV_Init(&argc, &argv, false);
   PV::Communicator *comm = pv_init->getCommunicator();
   int status             = PV_SUCCESS;
   if (testOneToOneShared(comm) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testManyToOneShared(comm) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testOneToManyShared(comm) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testOneToOneNonshared(comm) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testManyToOneNonshared(comm) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (testOneToManyNonshared(comm) != PV_SUCCESS) {
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed on rank " << comm->globalCommRank() << ".\n";
   }
   delete pv_init;
   return status;
}

void checkSizeCompatibility(PVLayerLoc const &loc, PV::Communicator *comm) {
   FatalIf(
         loc.nx * comm->numCommColumns() != loc.nxGlobal,
         "testOneToOneShared: Number of process in the x-direction was %d, but it must be a "
         "divisor of %d\n",
         comm->numCommColumns(),
         loc.nxGlobal);
   FatalIf(
         loc.ny * comm->numCommRows() != loc.nyGlobal,
         "testOneToOneShared: Number of process in the y-direction was %d, but it must be a "
         "divisor of %d\n",
         comm->numCommRows(),
         loc.nyGlobal);
}

int testOneToOneShared(PV::Communicator *comm) {
   int const nxPre      = 8;
   int const nyPre      = 8;
   int const nfPre      = 3;
   int const nxPost     = 8;
   int const nyPost     = 8;
   int const nfPost     = 8;
   int const patchSizeX = 3;
   int const patchSizeY = 3;
   return TestShared(
         std::string("OneToOneShared"),
         nxPre,
         nyPre,
         nfPre,
         nxPost,
         nyPost,
         nfPost,
         patchSizeX,
         patchSizeY,
         comm);
}

int testManyToOneShared(PV::Communicator *comm) {
   int const nxPre      = 8;
   int const nyPre      = 8;
   int const nfPre      = 3;
   int const nxPost     = 4;
   int const nyPost     = 4;
   int const nfPost     = 8;
   int const patchSizeX = 3;
   int const patchSizeY = 3;
   return TestShared(
         std::string("ManyToOneShared"),
         nxPre,
         nyPre,
         nfPre,
         nxPost,
         nyPost,
         nfPost,
         patchSizeX,
         patchSizeY,
         comm);
}

int testOneToManyShared(PV::Communicator *comm) {
   int const nxPre      = 4;
   int const nyPre      = 4;
   int const nfPre      = 8;
   int const nxPost     = 8;
   int const nyPost     = 8;
   int const nfPost     = 3;
   int const patchSizeX = 6;
   int const patchSizeY = 6;
   return TestShared(
         std::string("OneToManyShared"),
         nxPre,
         nyPre,
         nfPre,
         nxPost,
         nyPost,
         nfPost,
         patchSizeX,
         patchSizeY,
         comm);
}

int testOneToOneNonshared(PV::Communicator *comm) {
   int const nxPre      = 8;
   int const nyPre      = 8;
   int const nfPre      = 3;
   int const nxPost     = 8;
   int const nyPost     = 8;
   int const nfPost     = 8;
   int const patchSizeX = 3;
   int const patchSizeY = 3;
   return TestNonshared(
         std::string("OneToOneNonshared"),
         nxPre,
         nyPre,
         nfPre,
         nxPost,
         nyPost,
         nfPost,
         patchSizeX,
         patchSizeY,
         comm);
}

int testManyToOneNonshared(PV::Communicator *comm) {
   int const nxPre      = 8;
   int const nyPre      = 8;
   int const nfPre      = 3;
   int const nxPost     = 4;
   int const nyPost     = 4;
   int const nfPost     = 8;
   int const patchSizeX = 3;
   int const patchSizeY = 3;
   return TestNonshared(
         std::string("ManyToOneNonshared"),
         nxPre,
         nyPre,
         nfPre,
         nxPost,
         nyPost,
         nfPost,
         patchSizeX,
         patchSizeY,
         comm);
}

int testOneToManyNonshared(PV::Communicator *comm) {
   int const nxPre      = 4;
   int const nyPre      = 4;
   int const nfPre      = 8;
   int const nxPost     = 8;
   int const nyPost     = 8;
   int const nfPost     = 3;
   int const patchSizeX = 6;
   int const patchSizeY = 6;
   return TestNonshared(
         std::string("OneToManyNonshared"),
         nxPre,
         nyPre,
         nfPre,
         nxPost,
         nyPost,
         nfPost,
         patchSizeX,
         patchSizeY,
         comm);
}
