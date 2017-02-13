#include <sstream>
#include <stdio.h>

/**
 * Tests copying boundary regions over MPI using the BorderExchange class.
 *
 * This test does not create a HyPerCol, and all command line arguments
 * are ignored except --require-return.
 *
 * A PVLayerLoc struct is created for a local restricted size of 128x128x1
 * and margin width of 16, and an extended buffer (160x160x1) is created
 * in each MPI process.  The restricted part of the buffer is filled with
 * the global restricted index, and the border regions are zero.
 *
 * Then BorderExchange::exchange() is called with the buffer and the PVLayerLoc,
 * and the values in the extended region are checked.
 */

#undef DEBUG_PRINT

#include <columns/Communicator.hpp>
#include <columns/PV_Init.hpp>
#include <io/fileio.hpp>
#include <utils/BorderExchange.hpp>

static int check_borders(float *buf, PV::BorderExchange *borderExchanger, PVLayerLoc loc);

int main(int argc, char *argv[]) {
   PV::PV_Init *initObj   = new PV::PV_Init(&argc, &argv, true /*allowUnrecognizedArguments*/);
   PV::Communicator *comm = initObj->getCommunicator();
   PV::MPIBlock const *mpiBlock = comm->getLocalMPIBlock();

   PVLayerLoc loc;

   int nxProc = mpiBlock->getNumColumns();
   int nyProc = mpiBlock->getNumRows();

   int rowIndex    = mpiBlock->getRowIndex();
   int columnIndex = mpiBlock->getColumnIndex();

   loc.nbatch = 1;
   loc.nx     = 128;
   loc.ny     = 128;
   loc.nf     = 1;

   loc.nbatchGlobal = 1;
   loc.nxGlobal     = nxProc * loc.nx;
   loc.nyGlobal     = nyProc * loc.ny;

   // this info not used for send/recv
   loc.kb0 = 0;
   loc.kx0 = columnIndex * loc.nx;
   loc.ky0 = rowIndex * loc.ny;

   const int nxBorder = 16;
   loc.halo.lt        = nxBorder;
   loc.halo.rt        = nxBorder;
   const int nyBorder = 16;
   loc.halo.dn        = nyBorder;
   loc.halo.up        = nyBorder;

   int numItems            = (2 * nxBorder + loc.nx) * (2 * nyBorder + loc.ny);

   // create a local portion of the "image"
   float *image = new float[numItems];

   int k0 = columnIndex * loc.nx + rowIndex * loc.ny * loc.nxGlobal;
   int sy = 2 * nxBorder + loc.nx;

   for (int ky = 0; ky < loc.ny; ky++) {
      int k      = k0 + ky * loc.nxGlobal;
      float *buf = image + nxBorder + (ky + nyBorder) * sy;
      for (int kx = 0; kx < loc.nx; kx++) {
         buf[kx] = (float)k++;
      }
   }

   // send and recv the "image"
   PV::BorderExchange *borderExchanger = new PV::BorderExchange(*mpiBlock, loc);
   InfoLog().printf(
         "[%d]: nxProc==%d nyProc==%d rowIndex==%d columnIndex==%d numNeighbors==%d\n",
         mpiBlock->getRank(),
         nxProc,
         nyProc,
         rowIndex,
         columnIndex,
         borderExchanger->getNumNeighbors());
   InfoLog().flush();

   std::vector<MPI_Request> req;
   borderExchanger->exchange(image, req);
   int err = borderExchanger->wait(req);
   if (err != 0) {
      ErrorLog().printf("[%d]: BorderExchange::wait failed\n", mpiBlock->getRank());
   }

   if (err == 0) {
      err = check_borders(image, borderExchanger, loc);
      if (err != 0) {
         ErrorLog().printf("[%d]: check_borders failed\n", mpiBlock->getRank());
      }
      else {
         InfoLog().printf("[%d]: check_borders succeeded\n", mpiBlock->getRank());
      }
   }

   delete borderExchanger;
   delete initObj;

   return err;
}

static int check_borders(float *image, PV::BorderExchange *borderExchanger, PVLayerLoc loc) {
   int err = 0;

   const int nx       = (int)loc.nx;
   const int ny       = (int)loc.ny;
   const PVHalo *halo = &loc.halo;

   PV::MPIBlock const *mpiBlock = borderExchanger->getMPIBlock();

   const int rowIndex    = mpiBlock->getRowIndex();
   const int columnIndex = mpiBlock->getColumnIndex();
   int const numRows     = mpiBlock->getNumRows();
   int const numColumns  = mpiBlock->getNumColumns();

   int k0   = columnIndex * nx + rowIndex * ny * loc.nxGlobal;
   int sy   = nx + halo->lt + halo->rt;
   int rank = mpiBlock->getRank();

   // northwest
   // Note that if this MPI process is on the northern edge of the MPI quilt,
   // the northwestern neighbor is the same as the western neighbor, and the
   // the data sent will be from the extended region of the neighbor and
   // hence all zeroes.  Only if the northwest neighbor is distinct from the
   // northern neighbor and from the western neighbor will the data the neighbor
   // sent be from the restricted region.
   if (borderExchanger->hasNorthwesternNeighbor(rowIndex, columnIndex, numRows, numColumns)) {
      int northwestIndex = borderExchanger->northwest(rowIndex, columnIndex, numRows, numColumns);
      int northIndex     = borderExchanger->north(rowIndex, columnIndex, numRows, numColumns);
      int westIndex      = borderExchanger->west(rowIndex, columnIndex, numRows, numColumns);
      if (northwestIndex != northIndex && northwestIndex != westIndex) {
         for (int ky = 0; ky < halo->up; ky++) {
            int k      = (k0 - halo->lt) + (ky - halo->up) * loc.nxGlobal;
            float *buf = image + ky * sy;
            for (int kx = 0; kx < halo->lt; kx++) {
               if ((int)buf[kx] != k++) {
                  ErrorLog().printf(
                        "[?]: northwest check_borders failed kx==%d ky==%d observed==%f "
                        "correct==%d addr==%p\n",
                        kx,
                        ky,
                        (double)buf[kx],
                        k - 1,
                        &buf[kx]);
                  return 1;
               }
            }
         }
      }
      else {
         for (int ky = 0; ky < halo->up; ky++) {
            float *buf = image + ky * sy;
            for (int kx = 0; kx < halo->lt; kx++) {
               if ((int)buf[kx] != 0) {
                  ErrorLog().printf(
                        "[?]: northwest check_borders failed kx==%d ky==%d observed==%f correct==0 "
                        "addr==%p\n",
                        kx,
                        ky,
                        (double)buf[kx],
                        &buf[kx]);
                  return 1;
               }
            }
         }
      }
   }

   // west
   if (borderExchanger->hasWesternNeighbor(rowIndex, columnIndex, numRows, numColumns)) {
      for (int ky = 0; ky < ny; ky++) {
         int k      = k0 - halo->up + ky * loc.nxGlobal;
         float *buf = image + (ky + halo->up) * sy;
         for (int kx = 0; kx < halo->lt; kx++) {
            if ((int)buf[kx] != k++) {
               ErrorLog().printf(
                     "[?]: check_borders failed kx==%d ky==%d k0==%d observed==%f correct==%d "
                     "addr==%p\n",
                     kx,
                     ky,
                     k0,
                     (double)buf[kx],
                     k - 1,
                     &buf[kx]);
               return 1;
            }
         }
      }
   }

   // east
   if (borderExchanger->hasEasternNeighbor(rowIndex, columnIndex, numRows, numColumns)) {
      for (int ky = 0; ky < ny; ky++) {
         int k      = k0 + nx + ky * loc.nxGlobal;
         float *buf = image + (nx + halo->lt) + (ky + halo->up) * sy;
         for (int kx = 0; kx < halo->rt; kx++) {
            if ((int)buf[kx] != k++) {
               ErrorLog().printf(
                     "[?]: check_borders failed kx==%d ky==%d k0==%d observed==%f correct==%d "
                     "addr==%p\n",
                     kx,
                     ky,
                     k0,
                     (double)buf[kx],
                     k - 1,
                     &buf[kx]);
               return 1;
            }
         }
      }
   }

   // TODO: north, south, and other three corners.

   return err;
}
