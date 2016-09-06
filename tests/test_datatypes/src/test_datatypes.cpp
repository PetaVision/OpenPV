#include <stdio.h>
#include <sstream>

/**
 * Tests copying boundary regions over MPI using Communicator::exchange()
 *
 * This test does not create a HyPerCol, and all command line arguments
 * are ignored except --require-return.
 *
 * A PVLayerLoc struct is created for a local restricted size of 128x128x1
 * and margin width of 16, and an extended buffer (160x160x1) is created
 * in each MPI process.  The restricted part of the buffer is filled with
 * the global restricted index, and the border regions are zero.
 *
 * Then Communicator::exchange() is called with the buffer and the PVLayerLoc,
 * and the values in the extended region are checked.
 */

#undef DEBUG_PRINT

#include <columns/Communicator.hpp>
#include <columns/PV_Init.hpp>
#include <io/fileio.hpp>

static int check_borders(pvdata_t * buf, PV::Communicator * comm, PVLayerLoc loc);

int main(int argc, char * argv[])
{
   PV::PV_Init* initObj = new PV::PV_Init(&argc, &argv, true/*allowUnrecognizedArguments*/);
   PV::Communicator * comm = initObj->getCommunicator();
   
   // Handling of requireReturn copied from HyPerCol::initialize, since this test doesn't create a HyPerCol.
   if (initObj->getRequireReturnFlag()) {
      if (comm->commRank()==0) {
         fprintf(stdout, "Hit enter to begin! ");
         fflush(stdout);
         int charhit = -1;
         while(charhit != '\n') {
            charhit = getc(stdin);
         }
      }
      MPI_Barrier(comm->globalCommunicator());
   }
   int err = 0;
   PVLayerLoc loc;

   int nxProc = comm->numCommColumns();
   int nyProc = comm->numCommRows();

   int commRow = comm->commRow();
   int commCol = comm->commColumn();

   pvInfo().printf("[%d]: nxProc==%d nyProc==%d commRow==%d commCol==%d numNeighbors==%d\n", comm->commRank(), nxProc, nyProc, commRow, commCol, comm->numberOfNeighbors());
   pvInfo().flush();

   loc.nbatch = 1;
   loc.nx = 128;
   loc.ny = 128;
   loc.nf = 1;

   loc.nbatchGlobal = 1;
   loc.nxGlobal = nxProc * loc.nx;
   loc.nyGlobal = nyProc * loc.ny;

   // this info not used for send/recv
   loc.kb0 = 0; loc.kx0 = commCol*loc.nx; loc.ky0 = commRow*loc.ny;
   
   const int nxBorder = 16; loc.halo.lt = nxBorder; loc.halo.rt = nxBorder;
   const int nyBorder = 16; loc.halo.dn = nyBorder; loc.halo.up = nyBorder;

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
   std::vector<MPI_Request> req;
   if (err==0) {
      err = comm->exchange(image, datatypes, &loc, req);
      if (err != 0) {
         pvErrorNoExit().printf("[%d]: Communicator::exchange failed\n", comm->commRank());
      }
   }
   if (err==0) {
      err = comm->wait(req);
      if (err != 0) {
         pvErrorNoExit().printf("[%d]: Communicator::waitForExchange failed\n", comm->commRank());
      }
   }

   if (err==0) {
      err = check_borders(image, comm, loc);
      if (err != 0) {
         pvErrorNoExit().printf("[%d]: check_borders failed\n", comm->commRank());
      }
      else {
         pvInfo().printf("[%d]: check_borders succeeded\n", comm->commRank());
      }
   }

   comm->freeDatatypes(datatypes);

   delete initObj;

   return err;
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
   int rank = comm->commRank();

   // northwest
   // Note that if this MPI process is on the northern edge of the MPI quilt,
   // the northwestern neighbor is the same as the western neighbor, and the
   // the data sent will be from the extended region of the neighbor and
   // hence all zeroes.  Only if the northwest neighbor is distinct from the
   // northern neighbor and from the western neighbor will the data the neighbor
   // sent be from the restricted region.
   if (comm->hasNeighbor(PV::Communicator::NORTHWEST)) {
      if (comm->neighborIndex(rank, PV::Communicator::NORTHWEST) != comm->neighborIndex(rank, PV::Communicator::NORTH) &&
          comm->neighborIndex(rank, PV::Communicator::NORTHWEST) != comm->neighborIndex(rank, PV::Communicator::WEST)) {
         for (int ky = 0; ky < halo->up; ky++) {
            int k = (k0-halo->lt) + (ky-halo->up) * loc.nxGlobal;
            float * buf = image + ky * sy;
            for (int kx = 0; kx < halo->lt; kx++) {
               if ((int) buf[kx] != k++) {
                  pvErrorNoExit().printf("[?]: northwest check_borders failed kx==%d ky==%d observed==%f correct==%d addr==%p\n", kx, ky, buf[kx], k-1, &buf[kx]);
                  return 1;
               }
            }
         }
      }
      else {
         for (int ky = 0; ky < halo->up; ky++) {
            float * buf = image + ky * sy;
            for (int kx = 0; kx < halo->lt; kx++) {
               if ((int) buf[kx] != 0) {
                  pvErrorNoExit().printf("[?]: northwest check_borders failed kx==%d ky==%d observed==%f correct==0 addr==%p\n", kx, ky, buf[kx], &buf[kx]);
                  return 1;
               }
            }
         }
      }
   }

   // west
   if (comm->hasNeighbor(PV::Communicator::WEST)) {
      for (int ky = 0; ky < ny; ky++) {
         int k = k0 - halo->up + ky * loc.nxGlobal;
         float * buf = image + (ky + halo->up) * sy;
         for (int kx = 0; kx < halo->lt; kx++) {
            if ((int) buf[kx] != k++) {
               pvErrorNoExit().printf("[?]: check_borders failed kx==%d ky==%d k0==%d observed==%f correct==%d addr==%p\n", kx, ky, k0, buf[kx], k-1, &buf[kx]);
               return 1;
            }
         }
      }
   }

   // east
   if (comm->hasNeighbor(PV::Communicator::EAST)) {
      for (int ky = 0; ky < ny; ky++) {
         int k = k0 + nx + ky * loc.nxGlobal;
         float * buf = image + (nx + halo->lt) + (ky + halo->up) * sy;
         for (int kx = 0; kx < halo->rt; kx++) {
            if ((int) buf[kx] != k++) {
               pvErrorNoExit().printf("[?]: check_borders failed kx==%d ky==%d k0==%d observed==%f correct==%d addr==%p\n", kx, ky, k0, buf[kx], k-1, &buf[kx]);
               return 1;
            }
         }
      }
   }
   
   // TODO: north, south, and other three corners.

   return err;
}
