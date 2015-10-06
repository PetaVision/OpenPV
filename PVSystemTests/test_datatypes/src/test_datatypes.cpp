#include <stdio.h>

/**
 * This file tests copying to boundary regions (no boundary conditions)
 * using MPI.
 */

#undef DEBUG_PRINT

#include <columns/Communicator.hpp>
#include <columns/PV_Init.hpp>

static int check_borders(pvdata_t * buf, PV::Communicator * comm, PVLayerLoc loc);

int main(int argc, char * argv[])
{
   PV::PV_Init* initObj = new PV::PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   int err = 0;
   PVLayerLoc loc;

   PV::Communicator * comm = new PV::Communicator(initObj->getArguments());

   int nxProc = comm->numCommColumns();
   int nyProc = comm->numCommRows();

   int commRow = comm->commRow();
   int commCol = comm->commColumn();

   printf("[%d]: nxProc==%d nyProc==%d commRow==%d commCol==%d numNeighbors==%d\n", comm->commRank(), nxProc, nyProc, commRow, commCol, comm->numberOfNeighbors());  fflush(stdout);

   loc.nx = 128;
   loc.ny = 128;

   loc.nxGlobal = nxProc * loc.nx;
   loc.nyGlobal = nyProc * loc.ny;

   // this info not used for send/recv
   loc.kx0 = 0; loc.ky0 = 0;

   const int nxBorder = 16;
   const int nyBorder = 16;

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

   // send and recv the "image"

   comm->exchange(image, datatypes, &loc);

   err = check_borders(image, comm, loc);
   if (err != 0) {
      printf("[%d]: check_borders failed\n", comm->commRank());
   }
   else {
      printf("[%d]: check_borders succeeded\n", comm->commRank());
   }

   delete datatypes;
   delete comm;

   delete initObj;

   return 0;
}

static int check_borders(pvdata_t * image, PV::Communicator * comm, PVLayerLoc loc)
{
   int err = 0;

   const int nx = (int) loc.nx;
   const int ny = (int) loc.ny;
   const PVHalo * halo = &loc.halo;

   const int commRow = comm->commRow();
   const int commCol = comm->commColumn();

   int k0 = commCol * nx + commRow * ny * loc.nxGlobal;
   int sy = nx + halo->lt + halo->rt;

   // northwest
   if (comm->hasNeighbor(NORTHWEST)) {
      for (int ky = 0; ky < halo->up; ky++) {
         int k = k0 + ky * loc.nxGlobal;
         float * buf = image + ky * sy;
         for (int kx = 0; kx < halo->lt; kx++) {
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
         int k = k0 - halo->up + ky * loc.nxGlobal;
         float * buf = image + (ky + halo->up) * sy;
         for (int kx = 0; kx < halo->lt; kx++) {
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
         float * buf = image + (nx + halo->lt) + (ky + halo->up) * sy;
         for (int kx = 0; kx < halo->rt; kx++) {
            if ((int) buf[kx] != k++) {
               printf("[?]: check_borders failed kx==%d ky==%d k0==%d buf==%f k=%d addr==%p\n", kx, ky, k0, buf[kx], k-1, &buf[kx]);
               return 1;
            }
         }
      }
   }

   return err;
}
