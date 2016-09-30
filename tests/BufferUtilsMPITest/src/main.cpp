#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/PVLog.hpp"
#include "columns/PV_Arguments.hpp"
#include "columns/Communicator.hpp"
#include "arch/mpi/mpi.h"

#include <vector>

using PV::Buffer;
using PV::PV_Arguments;
using PV::Communicator;
using std::vector;

namespace BufferUtils = PV::BufferUtils;

// BufferSlicer::scatter(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
// BufferSlicer::gather(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
void testRestricted(int argc, char** argv) {
   PV_Arguments *args = new PV_Arguments(argc, argv, false);
   Communicator *comm = new Communicator(args); 
   int rank = comm->commRank();

   pvInfo() << "Setup complete on rank " << rank << ". Running test.\n";

   unsigned int sliceX = 4 / comm->numCommColumns();
   unsigned int sliceY = 4 / comm->numCommRows();

   Buffer<float> dataBuffer;

   // Send / receive the test data, depending on what rank we are
   vector<float> result;
   if (rank == 0) {
      vector<float> testData = {
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3
         };
      dataBuffer.set(testData, 4, 4, 1); 
      BufferUtils::scatter<float>(comm, dataBuffer, sliceX, sliceY);
   }
   else {
      dataBuffer.resize(sliceX, sliceY, 1);
      BufferUtils::scatter<float>(comm, dataBuffer, sliceX, sliceY);
   }
   result = dataBuffer.asVector();

   pvInfo() << "Scatter complete on rank " << rank << ".\n";

   // Check to make sure the chunk of data we received is correct
   if (comm->commSize() == 1) {
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
   else if (comm->commSize() == 2) {
      pvErrorIf(result.size() != 8,
         "Failed. Expected 8 values, found %d.\n", result.size());
      vector<float> expected = {
            0, 0, 1, 1,
            0, 0, 1, 1
         };
      for (size_t i = 0; i < result.size(); ++i) {
         // We expect 0s and 1s on rank 0, 2s and 3s on rank 1
         pvErrorIf(result.at(i) != rank * 2 + expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               rank * 2 + (int)expected.at(i), (int)result.at(i));
      }
   }
   else if (comm->commSize() == 4) {
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

   pvInfo() << "Beginning gather on rank " << rank << "\n";

   dataBuffer.set(BufferUtils::gather<float>(comm, dataBuffer, sliceX, sliceY));
   result = dataBuffer.asVector();

   if(rank == 0) {
      vector<float> expected = {
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3
         };
      pvErrorIf(result.size() != 16,
         "Failed. Expected 16 values, found %d.\n", result.size());
      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i), (int)result.at(i));
 
      }
   }
}

// BufferSlicer::scatter(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
// BufferSlicer::gather(Buffer &buffer, uint sliceStrideX, uint sliceStrideY)
void testExtended(int argc, char** argv) {
   PV_Arguments *args = new PV_Arguments(argc, argv, false);
   Communicator *comm = new Communicator(args); 
   int rank = comm->commRank();

   pvInfo() << "Setup complete on rank " << rank << ". Running test.\n";

   unsigned int sliceX = 4 / comm->numCommColumns();
   unsigned int sliceY = 4 / comm->numCommRows();

   Buffer<float> dataBuffer;

   // Send / receive the test data, depending on what rank we are
   vector<float> result;
   if (rank == 0) {
      vector<float> testData = {
            9, 9, 9, 9, 9, 9,
            9, 0, 0, 1, 1, 9,
            9, 0, 0, 1, 1, 9,
            9, 2, 2, 3, 3, 9,
            9, 2, 2, 3, 3, 9,
            9, 9, 9, 9, 9, 9
         };

      dataBuffer.set(testData, 6, 6, 1); 
      BufferUtils::scatter<float>(comm, dataBuffer, sliceX, sliceY);
   }
   else {
      // We have a 1 element margin on each side
      dataBuffer.resize(sliceX + 2, sliceY + 2, 1);
      BufferUtils::scatter<float>(comm, dataBuffer, sliceX, sliceY);
   }
   result = dataBuffer.asVector();

   pvInfo() << "Scatter complete on rank " << rank << ".\n";

   // Check to make sure the chunk of data we received is correct
   if (comm->commSize() == 1) {
      pvErrorIf(result.size() != 6 * 6,
         "Failed. Expected 36 values, found %d.\n", result.size());
      vector<float> expected = {
            9, 9, 9, 9, 9, 9,
            9, 0, 0, 1, 1, 9,
            9, 0, 0, 1, 1, 9,
            9, 2, 2, 3, 3, 9,
            9, 2, 2, 3, 3, 9,
            9, 9, 9, 9, 9, 9
          };
      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i), (int)result.at(i));
      }
   }
   else if (comm->commSize() == 2) {
      pvErrorIf(result.size() != 4 * 6,
         "Failed. Expected 24 values, found %d.\n", result.size());

      vector<float> expected; 
      switch (rank) {
         case 0:
            expected = {
               9, 9, 9, 9, 9, 9,
               9, 0, 0, 1, 1, 9,
               9, 0, 0, 1, 1, 9,
               9, 2, 2, 3, 3, 9
             };
            break;
         case 1:
            expected = {
               9, 0, 0, 1, 1, 9,
               9, 2, 2, 3, 3, 9,
               9, 2, 2, 3, 3, 9,
               9, 9, 9, 9, 9, 9
            };
            break;
         default:
            pvError() << "Invalid rank.\n";
      }

      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i), (int)result.at(i));
      }
   }
   else if (comm->commSize() == 4) {
      pvErrorIf(result.size() != 4 * 4,
         "Failed. Expected 16 values, found %d.\n", result.size());

      vector<float> expected; 
      switch (rank) {
         case 0:
            expected = {
               9, 9, 9, 9,
               9, 0, 0, 1,
               9, 0, 0, 1,
               9, 2, 2, 3
             };
            break;
         case 1:
            expected = {
               9, 9, 9, 9,
               0, 1, 1, 9,
               0, 1, 1, 9,
               2, 3, 3, 9 
            };
            break;
         case 2:
            expected = {
               9, 0, 0, 1,
               9, 2, 2, 3,
               9, 2, 2, 3,
               9, 9, 9, 9
            };
            break;
         case 3:
            expected = {
               0, 1, 1, 9,
               2, 3, 3, 9,
               2, 3, 3, 9,
               9, 9, 9, 9
            };
            break;
          default:
            pvError() << "Invalid rank.\n";
      }

      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i), (int)result.at(i));
      }
   }
   else {
      pvError() << "Failed. Must test using 1, 2, or 4 MPI processes.\n";
   }

   pvInfo() << "Beginning gather on rank " << rank << "\n";

   dataBuffer.set(BufferUtils::gather(comm, dataBuffer, sliceX, sliceY));
   result = dataBuffer.asVector();

   if(rank == 0) {
      vector<float> expected = {
            9, 9, 9, 9, 9, 9,
            9, 0, 0, 1, 1, 9,
            9, 0, 0, 1, 1, 9,
            9, 2, 2, 3, 3, 9,
            9, 2, 2, 3, 3, 9,
            9, 9, 9, 9, 9, 9
          };
      pvErrorIf(result.size() != 6 * 6,
         "Failed. Expected 36 values, found %d.\n", result.size());
      for (size_t i = 0; i < result.size(); ++i) {
         pvErrorIf(result.at(i) != expected.at(i),
            "Failed. Expected to find %d, found %d instead.\n",
               (int)expected.at(i), (int)result.at(i));
      }
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

   pvInfo() << "Rank " << rank << ": Testing restricted BufferUtils::scatter() and BufferUtils::gather():\n";
   testRestricted(5, args);
   pvInfo() << "Rank " << rank << ": Completed.\n";

   pvInfo() << "Rank " << rank << ": Testing extended BufferUtils::scatter() and BufferUtils::gather():\n";
   testExtended(5, args);
   pvInfo() << "Rank " << rank << ": Completed.\n";


   MPI_Finalize();

   free(args[1]);
   free(args[2]);
   free(args[3]);
   free(args[4]);

   pvInfo() << "BufferUtilsMPI tests completed successfully!\n";
   return EXIT_SUCCESS;
}
