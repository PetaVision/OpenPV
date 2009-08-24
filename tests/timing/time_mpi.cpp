/**
 * This file tests MPI send/recv time for layers.
 */

#undef DEBUG_PRINT

#include "clock.h"
#include "../../src/columns/Communicator.hpp"

int main(int argc, char * argv[])
{
   int err = 0;
   PVLayerLoc loc;

   const int nloops = 1000;

   PV::Communicator * comm = new PV::Communicator(&argc, &argv);

   const int rank = comm->commRank();

   const int nxProc = comm->numCommColumns();
   const int nyProc = comm->numCommRows();

   const int commRow = comm->commRow();
   const int commCol = comm->commColumn();

   if (rank == 0) {
      fprintf(stderr, "\n[0]: nxProc==%d nyProc==%d commRow==%d commCol==%d numNeighbors==%d\n\n", nxProc, nyProc, commRow, commCol, comm->numberOfNeighbors());
   }

   loc.nx = 128;
   loc.ny = 128;

   loc.nxGlobal = nxProc * loc.nx;
   loc.nyGlobal = nyProc * loc.ny;

   // this info not used for send/recv
   loc.kx0 = 0; loc.ky0 = 0;

   loc.nxBorder = 16;
   loc.nyBorder = 16;
   int numItems = (2*loc.nxBorder + loc.nx) * (2*loc.nyBorder + loc.ny);

   const int nxBorder = loc.nxBorder;
   const int nyBorder = loc.nyBorder;

   MPI_Datatype * datatypes = comm->newDatatypes(&loc);

   // create a local portion of the "image"
   float * image = new float [numItems];

   int k0 = commCol * loc.nx + commRow * loc.ny * loc.nxGlobal;
   int sy = 2 * loc.nxBorder + loc.nx;

   for (int ky = 0; ky < loc.ny; ky++) {
      int k = k0 + ky * loc.nxGlobal;
      float * buf = image + nxBorder + (ky + nyBorder) * sy;
      for (int kx = 0; kx < loc.nx; kx++) {
         buf[kx] = (float) k++;
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   start_clock();
   double start = MPI_Wtime();

   for (int n = 0; n < nloops; n++) {
      comm->send(image, datatypes, &loc);
      comm->recv(image, datatypes, &loc);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   stop_clock();
   double elapsed = MPI_Wtime() - start;

   if (rank == 0) {
      float cycle_time = (1000 * elapsed) / nloops;
      fprintf(stderr, "\n[0]: number of send/recv cycles == %d\n", nloops);
      fprintf(stderr, "[0]: time per send/recv cycle   == %f ms\n", cycle_time);
      fprintf(stderr, "[0]: elapsed time (MPI_Wtime)   == %f s\n\n", (float) elapsed);
   }

   delete datatypes;
   delete comm;

   return err;
}
