#include "utils/BufferSlicer.hpp"
#include "utils/PVLog.hpp"
#include "columns/PV_Arguments.hpp"
#include "columns/Communicator.hpp"
#include "arch/mpi/mpi.h"

#include <vector>

using PV::Buffer;
using PV::BufferSlicer;
using PV::PV_Arguments;
using PV::Communicator;
using std::vector;

PV_Arguments *args;

// BufferSlicer::scatter(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
void testScatterRestricted(int argc, char** argv) {
   PV_Arguments *args = new PV_Arguments(argc, argv, false);
   Communicator comm(args); 
   BufferSlicer slicer(comm);
   int rank = comm.commRank();

   pvInfo() << "Setup complete on rank " << rank << ". Running test.\n";

   unsigned int sliceX = 4 / comm.numCommColumns();
   unsigned int sliceY = 4 / comm.numCommRows();

   vector<float> result;
   if (rank == 0) {
      vector<float> testData = {
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3
         };

      Buffer send(testData, 4, 4, 1); 
      slicer.scatter(send, sliceX, sliceY);
      result = send.asVector();
   }
   else {
      Buffer recv(sliceX, sliceY, 1);
      slicer.scatter(recv, sliceX, sliceY);
      result = recv.asVector();
   }

   pvInfo() << "Scatter complete on rank " << rank << ".\n";

   if (comm.commSize() == 1) {
      pvErrorIf(result.size() != 16,
         "Failed. Expected 16 values, found %d.\n", result.size());
      vector<float> expected = {
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3
         };
      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i), (int)result.at(i));
      }
   }
   else if (comm.commSize() == 2) {
      pvErrorIf(result.size() != 8,
         "Failed. Expected 8 values, found %d.\n", result.size());
      vector<float> expected = {
            0, 0, 1, 1,
            0, 0, 1, 1
         };
      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != rank * 2 + expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               rank * 2 + (int)expected.at(i), (int)result.at(i));
      }
   }
   else if (comm.commSize() == 4) {
      pvErrorIf(result.size() != 4,
         "Failed. Expected 4 values, found %d.\n", result.size());
      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != rank,
            "Failed. Expected to find %d, found %d instead.\n",
               rank, (int)result.at(i));
      }
   }
   else {
      pvError() << "Failed. Must test using 1, 2, or 4 MPI processes.\n";
   }
}

int main(int argc, char** argv) {
   int numProcs = -1;
   int rank     = -1;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   assert(rank != -1);
   assert(numProcs == 1 || numProcs == 2 || numProcs == 4);

   char* args[5];
   args[0] = argv[0];
   args[1] = strdup("-rows");
   args[2] = numProcs == 1
           ? strdup("1")
           : strdup("2");
   args[3] = strdup("-columns");
   args[4] = numProcs != 4
           ? strdup("1")
           : strdup("2");

   pvInfo() << "Testing restricted BufferSlicer::scatter():\n";
   testScatterRestricted(3, args);
   pvInfo() << "Completed.\n";

   MPI_Finalize();

   free(args[1]);
   free(args[2]);
   free(args[3]);
   free(args[4]);

   pvInfo() << "BufferSlicer tests completed successfully!\n";
   return EXIT_SUCCESS;
}
