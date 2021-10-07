#include "WaitForReturn.hpp"

#include <cstdio>

namespace PV {

void WaitForReturn(MPI_Comm comm) {
   int rank;
   MPI_Comm_rank(comm, &rank);
   std::fflush(stdout);
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
