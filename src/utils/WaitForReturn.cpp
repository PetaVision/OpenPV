#include "WaitForReturn.hpp"

#include <cstdio>
#include <unistd.h>

namespace PV {

void printRankAndPid(MPI_Comm comm) {
   int rank;
   MPI_Comm_rank(comm, &rank);
   int size;
   MPI_Comm_size(comm, &size);
   for (int r=0; r < size; ++r) {
      if (r==rank) {
         std::printf("Rank %d, pid %d\n", rank, static_cast<int>(getpid()));
         std::fflush(stdout);
      }
      MPI_Barrier(comm);
   }
}

void WaitForReturn(MPI_Comm comm) {
   int rank;
   MPI_Comm_rank(comm, &rank);
   printRankAndPid(comm);
   MPI_Barrier(comm);
   if (rank == 0) {
      std::printf("Hit enter to begin! ");
      std::fflush(stdout);
      int charhit = -1;
      while (charhit != '\n') {
         charhit = std::getc(stdin);
      }
   }
   MPI_Barrier(comm);
}

} // namespace PV
