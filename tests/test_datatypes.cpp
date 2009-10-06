#include <stdio.h>

/**
 * This file tests copying to boundary regions (no boundary conditions)
 * using MPI.
 */

#undef DEBUG_PRINT

#include "../src/columns/Communicator.hpp"

static int check_borders(pvdata_t * buf, PV::Communicator * comm, LayerLoc loc);

int main(int argc, char * argv[])
{
   int err = 0;
   LayerLoc loc;

   PV::Communicator * comm = new PV::Communicator(&argc, &argv);

   int nxProc = comm->numCommColumns();
   int nyProc = comm->numCommRows();

   int commRow = comm->commRow();
   int commCol = comm->commColumn();

   printf("[%d]: nxProc==%d nyProc==%d commRow==%d commCol==%d numNeighbors==%d\n", comm->commRank(), nxProc, nyProc, commRow, commCol, comm->numberOfNeighbors());  fflush(stdout);

   //   for (int n = 0; n < MAX_NEIGHBORS+1; n++) {
   //      printf("[%d]: hasNeighbor(%d)=%d\n", comm->commRank(), n, comm->hasNeighbor(n));  fflush(stdout);
   //   }

   loc.nx = 128;
   loc.ny = 128;

   loc.nxGlobal = nxProc * loc.nx;
   loc.nyGlobal = nyProc * loc.ny;

   // this info not used for send/recv
   loc.kx0 = 0; loc.ky0 = 0;

   loc.nPad = 16;

   const int nxBorder = loc.nPad;
   const int nyBorder = loc.nPad;

   int numItems = (2*nxBorder + loc.nx) * (2*nyBorder + loc.ny);


   MPI_Datatype * datatypes = comm->newDatatypes(&loc);

   // create a local portion of the "image"
   float * image = new float [numItems];

   int k0 = commCol * loc.nx + commRow * loc.ny * loc.nxGlobal;
   int sy = 2 * nxBorder + loc.nx;

   for (int ky = 0; ky < loc.ny; ky++) {
      int k = k0 + ky * loc.nxGlobal;
      float * buf = image + nxBorder + (ky + nyBorder) * sy;
      for (int kx = 0; kx < loc.nx; kx++) {
         buf[kx] = (float) k++;
      }
   }

   //   printf("[%d]: k0==%d image[2579]==%f addr==%p\n", comm->commRank(), k0, image[2579], &image[2579]);  fflush(stdout);

   // send and recv the "image"

   //   printf("[%d]: exchanging, k0==%d\n", comm->commRank(), k0);  fflush(stdout);
   comm->exchange(image, datatypes, &loc);

   //   printf("[%d]: border check...\n", comm->commRank());  fflush(stdout);
   err = check_borders(image, comm, loc);
   if (err != 0) {
      printf("[%d]: check_borders failed\n", comm->commRank());
   }
   else {
      printf("[%d]: check_borders succeeded\n", comm->commRank());
   }

   delete datatypes;
   delete comm;

   return 0;
}

static int check_borders(pvdata_t * image, PV::Communicator * comm, LayerLoc loc)
{
   int err = 0;

   const int nx = (int) loc.nx;
   const int ny = (int) loc.ny;

   const int nxBorder = loc.nPad;
   const int nyBorder = loc.nPad;

   const int commRow = comm->commRow();
   const int commCol = comm->commColumn();

   int k0 = commCol * nx + commRow * ny * loc.nxGlobal;
   int sy = 2 * nxBorder + nx;

   // northwest
   if (comm->hasNeighbor(NORTHWEST)) {
      for (int ky = 0; ky < nyBorder; ky++) {
         int k = k0 + ky * loc.nxGlobal;
         float * buf = image + ky * sy;
         for (int kx = 0; kx < nxBorder; kx++) {
            if ((int) buf[kx] != k++) {
               printf("[?]: check_borders failed kx==%d ky==%d buf==%f k=%d addr==%p\n", kx, ky, buf[kx], k-1, &buf[kx]);
               return 1;
            }
         }
      }
   }

   // west
   if (comm->hasNeighbor(WEST)) {
      for (int ky = 0; ky < ny; ky++) {
         int k = k0 - nxBorder + ky * loc.nxGlobal;
         float * buf = image + (ky + nyBorder) * sy;
         for (int kx = 0; kx < nxBorder; kx++) {
            if ((int) buf[kx] != k++) {
               printf("[?]: check_borders failed kx==%d ky==%d k0==%d buf==%f k=%d addr==%p\n", kx, ky, k0, buf[kx], k-1, &buf[kx]);
               return 1;
            }
         }
      }
   }

   // east
   if (comm->hasNeighbor(EAST)) {
      for (int ky = 0; ky < ny; ky++) {
         int k = k0 + nx + ky * loc.nxGlobal;
         float * buf = image + (nx + nxBorder) + (ky + nyBorder) * sy;
         for (int kx = 0; kx < nxBorder; kx++) {
            if ((int) buf[kx] != k++) {
               printf("[?]: check_borders failed kx==%d ky==%d k0==%d buf==%f k=%d addr==%p\n", kx, ky, k0, buf[kx], k-1, &buf[kx]);
               return 1;
            }
         }
      }
   }

   return err;
}
