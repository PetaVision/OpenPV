#include "arch/mpi/mpi.h"
#include "columns/CommandLineArguments.hpp"
#include "columns/Communicator.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/PVLog.hpp"

#include <vector>

using PV::Buffer;
using PV::CommandLineArguments;
using PV::Communicator;
using std::vector;

namespace BufferUtils = PV::BufferUtils;

// BufferSlicer::scatter(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
// BufferSlicer::gather(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
void testRestricted(Communicator const *comm) {
   int rank = comm->commRank();

   InfoLog() << "Setup complete on rank " << rank << ". Running test.\n";

   unsigned int sliceX = 4 / comm->numCommColumns();
   unsigned int sliceY = 4 / comm->numCommRows();

   Buffer<float> dataBuffer;

   // Send / receive the test data, depending on what rank we are
   vector<float> result;
   if (rank == 0) {
      vector<float> testData = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
      dataBuffer.set(testData, 4, 4, 1);
      BufferUtils::scatter<float>(comm->getLocalMPIBlock(), dataBuffer, sliceX, sliceY, 0, 0);
   }
   else {
      dataBuffer.resize(sliceX, sliceY, 1);
      BufferUtils::scatter<float>(comm->getLocalMPIBlock(), dataBuffer, sliceX, sliceY, 0, 0);
   }
   result = dataBuffer.asVector();

   InfoLog() << "Scatter complete on rank " << rank << ".\n";

   // Check to make sure the chunk of data we received is correct
   if (comm->commSize() == 1) {
      FatalIf(result.size() != 16, "Failed. Expected 16 values, found %d.\n", result.size());
      vector<float> expected = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
      for (size_t i = 0; i < result.size(); ++i) {
         FatalIf(
               result.at(i) != expected.at(i),
               "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i),
               (int)result.at(i));
      }
   }
   else if (comm->commSize() == 2) {
      FatalIf(result.size() != 8, "Failed. Expected 8 values, found %d.\n", result.size());
      vector<float> expected = {0, 0, 1, 1, 0, 0, 1, 1};
      for (size_t i = 0; i < result.size(); ++i) {
         // We expect 0s and 1s on rank 0, 2s and 3s on rank 1
         FatalIf(
               result.at(i) != rank * 2 + expected.at(i),
               "Failed. Expected to find %d, found %d instead.\n",
               rank * 2 + (int)expected.at(i),
               (int)result.at(i));
      }
   }
   else if (comm->commSize() == 4) {
      FatalIf(result.size() != 4, "Failed. Expected 4 values, found %d.\n", result.size());
      for (size_t i = 0; i < result.size(); ++i) {
         FatalIf(
               result.at(i) != rank,
               "Failed. Expected to find %d, found %d instead.\n",
               rank,
               (int)result.at(i));
      }
   }
   else {
      Fatal() << "Failed. Must test using 1, 2, or 4 MPI processes.\n";
   }

   InfoLog() << "Beginning gather on rank " << rank << "\n";

   dataBuffer.set(
         BufferUtils::gather<float>(comm->getLocalMPIBlock(), dataBuffer, sliceX, sliceY, 0, 0));
   result = dataBuffer.asVector();

   if (rank == 0) {
      vector<float> expected = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
      FatalIf(result.size() != 16, "Failed. Expected 16 values, found %d.\n", result.size());
      for (size_t i = 0; i < result.size(); ++i) {
         FatalIf(
               result.at(i) != expected.at(i),
               "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i),
               (int)result.at(i));
      }
   }
}

// BufferSlicer::scatter(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
// BufferSlicer::gather(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
void testExtended(Communicator const *comm) {
   int rank = comm->commRank();

   InfoLog() << "Setup complete on rank " << rank << ". Running test.\n";

   unsigned int sliceX = 4 / comm->numCommColumns();
   unsigned int sliceY = 4 / comm->numCommRows();

   Buffer<float> dataBuffer;

   // Send / receive the test data, depending on what rank we are
   vector<float> result;
   if (rank == 0) {
      vector<float> testData = {9, 9, 9, 9, 9, 9, 9, 0, 0, 1, 1, 9, 9, 0, 0, 1, 1, 9,
                                9, 2, 2, 3, 3, 9, 9, 2, 2, 3, 3, 9, 9, 9, 9, 9, 9, 9};

      dataBuffer.set(testData, 6, 6, 1);
      BufferUtils::scatter<float>(comm->getLocalMPIBlock(), dataBuffer, sliceX, sliceY, 0, 0);
   }
   else {
      // We have a 1 element margin on each side
      dataBuffer.resize(sliceX + 2, sliceY + 2, 1);
      BufferUtils::scatter<float>(comm->getLocalMPIBlock(), dataBuffer, sliceX, sliceY, 0, 0);
   }
   result = dataBuffer.asVector();

   InfoLog() << "Scatter complete on rank " << rank << ".\n";

   // Check to make sure the chunk of data we received is correct
   if (comm->commSize() == 1) {
      FatalIf(result.size() != 6 * 6, "Failed. Expected 36 values, found %d.\n", result.size());
      vector<float> expected = {9, 9, 9, 9, 9, 9, 9, 0, 0, 1, 1, 9, 9, 0, 0, 1, 1, 9,
                                9, 2, 2, 3, 3, 9, 9, 2, 2, 3, 3, 9, 9, 9, 9, 9, 9, 9};
      for (size_t i = 0; i < result.size(); ++i) {
         FatalIf(
               result.at(i) != expected.at(i),
               "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i),
               (int)result.at(i));
      }
   }
   else if (comm->commSize() == 2) {
      FatalIf(result.size() != 4 * 6, "Failed. Expected 24 values, found %d.\n", result.size());

      vector<float> expected;
      switch (rank) {
         case 0:
            expected = {9, 9, 9, 9, 9, 9, 9, 0, 0, 1, 1, 9, 9, 0, 0, 1, 1, 9, 9, 2, 2, 3, 3, 9};
            break;
         case 1:
            expected = {9, 0, 0, 1, 1, 9, 9, 2, 2, 3, 3, 9, 9, 2, 2, 3, 3, 9, 9, 9, 9, 9, 9, 9};
            break;
         default: Fatal() << "Invalid rank.\n";
      }

      for (size_t i = 0; i < result.size(); ++i) {
         FatalIf(
               result.at(i) != expected.at(i),
               "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i),
               (int)result.at(i));
      }
   }
   else if (comm->commSize() == 4) {
      FatalIf(result.size() != 4 * 4, "Failed. Expected 16 values, found %d.\n", result.size());

      vector<float> expected;
      switch (rank) {
         case 0: expected = {9, 9, 9, 9, 9, 0, 0, 1, 9, 0, 0, 1, 9, 2, 2, 3}; break;
         case 1: expected = {9, 9, 9, 9, 0, 1, 1, 9, 0, 1, 1, 9, 2, 3, 3, 9}; break;
         case 2: expected = {9, 0, 0, 1, 9, 2, 2, 3, 9, 2, 2, 3, 9, 9, 9, 9}; break;
         case 3: expected = {0, 1, 1, 9, 2, 3, 3, 9, 2, 3, 3, 9, 9, 9, 9, 9}; break;
         default: Fatal() << "Invalid rank.\n";
      }

      for (size_t i = 0; i < result.size(); ++i) {
         FatalIf(
               result.at(i) != expected.at(i),
               "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i),
               (int)result.at(i));
      }
   }
   else {
      Fatal() << "Failed. Must test using 1, 2, or 4 MPI processes.\n";
   }

   InfoLog() << "Beginning gather on rank " << rank << "\n";

   dataBuffer.set(BufferUtils::gather(comm->getLocalMPIBlock(), dataBuffer, sliceX, sliceY, 0, 0));
   result = dataBuffer.asVector();

   if (rank == 0) {
      vector<float> expected = {9, 9, 9, 9, 9, 9, 9, 0, 0, 1, 1, 9, 9, 0, 0, 1, 1, 9,
                                9, 2, 2, 3, 3, 9, 9, 2, 2, 3, 3, 9, 9, 9, 9, 9, 9, 9};
      FatalIf(result.size() != 6 * 6, "Failed. Expected 36 values, found %d.\n", result.size());
      for (size_t i = 0; i < result.size(); ++i) {
         FatalIf(
               result.at(i) != expected.at(i),
               "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i),
               (int)result.at(i));
      }
   }
}

int main(int argc, char **argv) {
   int numProcs = -1;
   int rank     = -1;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   assert(rank != -1);
   assert(numProcs == 1 || numProcs == 2 || numProcs == 4);

   CommandLineArguments *args = new CommandLineArguments(argc, argv, false);
   args->setIntegerArgument("NumRows", numProcs == 1 ? 1 : 2);
   args->setIntegerArgument("NumColumns", numProcs != 4 ? 1 : 2);
   Communicator const *comm = new Communicator(args);

   InfoLog() << "Rank " << rank
             << ": Testing restricted BufferUtils::scatter() and BufferUtils::gather():\n";
   testRestricted(comm);
   InfoLog() << "Rank " << rank << ": Completed.\n";

   InfoLog() << "Rank " << rank
             << ": Testing extended BufferUtils::scatter() and BufferUtils::gather():\n";
   testExtended(comm);
   InfoLog() << "Rank " << rank << ": Completed.\n";

   delete comm;
   delete args;

   MPI_Finalize();

   InfoLog() << "BufferUtilsMPI tests completed successfully!\n";
   return EXIT_SUCCESS;
}
