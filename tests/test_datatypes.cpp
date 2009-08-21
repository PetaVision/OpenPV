/**
 * This file tests copying to boundary regions (no boundary conditions)
 * using MPI.
 */

#undef DEBUG_PRINT

#include "../src/columns/InterColComm.hpp"

static int check_borders(pvdata_t * buf, PV::InterColComm * ic, PVLayerLoc loc);

int main(int argc, char * argv[])
{
   int err = 0;
   PVLayerLoc loc;

   PV::InterColComm * ic = new PV::InterColComm(&argc, &argv);

   int nxProc = ic->numCommColumns();
   int nyProc = ic->numCommRows();

   int commRow = ic->commRow(ic->commRank());
   int commCol = ic->commColumn(ic->commRank());

   printf("[%d]: nxProc==%d nyProc==%d commRow==%d commCol==%d\n", ic->commRank(), nxProc, nyProc, commRow, commCol);  fflush(stdout);

   //   for (int n = 0; n < MAX_NEIGHBORS+1; n++) {
   //      printf("[%d]: hasNeighbor(%d)=%d\n", ic->commRank(), n, ic->hasNeighbor(n));  fflush(stdout);
   //   }

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

   MPI_Datatype * datatypes = ic->newDatatypes(&loc);

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

   //   printf("[%d]: k0==%d image[2579]==%f addr==%p\n", ic->commRank(), k0, image[2579], &image[2579]);  fflush(stdout);

   // send and recv the "image"

   //   printf("[%d]: sending, k0==%d\n", ic->commRank(), k0);  fflush(stdout);
   ic->send(image, datatypes, &loc);

   //   printf("[%d]: receiving...\n", ic->commRank());  fflush(stdout);
   ic->recv(image, datatypes, &loc);

   //   printf("[%d]: border check...\n", ic->commRank());  fflush(stdout);
   err = check_borders(image, ic, loc);
   if (err != 0) {
      printf("[%d]: check_borders failed\n", ic->commRank());
   }
   else {
      printf("[%d]: check_borders succeeded\n", ic->commRank());
   }

   delete datatypes;
   delete ic;

   return 0;
}

static int check_borders(pvdata_t * image, PV::InterColComm * ic, PVLayerLoc loc)
{
   int err = 0;

   const int rank = ic->commRank();

   const int nx = (int) loc.nx;
   const int ny = (int) loc.ny;

   const int nxBorder = loc.nxBorder;
   const int nyBorder = loc.nyBorder;

   const int commRow = ic->commRow(rank);
   const int commCol = ic->commColumn(rank);

   int k0 = commCol * nx + commRow * ny * loc.nxGlobal;
   int sy = 2 * loc.nxBorder + nx;

   // northwest
   if (ic->hasNeighbor(NORTHWEST)) {
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
   if (ic->hasNeighbor(WEST)) {
      for (int ky = 0; ky < ny; ky++) {
         int k = k0 - loc.nxBorder + ky * loc.nxGlobal;
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
   if (ic->hasNeighbor(EAST)) {
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
