/*
 * test_image_io.cpp
 *
 *  Created on: jan 2, 2010
 *      Author: rasmussn
 */

#include "../src/columns/HyPerCol.hpp"
#include "../src/connections/HyPerConn.hpp"
#include "../src/io/fileio.hpp"
#include <assert.h>
#include <stdio.h>

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif

#undef DEBUG_OUTPUT

using namespace PV;

PVPatch ** init_weight_patches(Communicator * comm, const PVLayerLoc * loc, int nf,
                               int nxp, int nyp, int nfp, bool zero_flag);
int test_output(Communicator * comm, const PVLayerLoc * loc, int nf,
                PVPatch ** inPatches, PVPatch ** outPatches);

const char filename[] = "output/test_weights_io.pvp";

int main(int argc, char* argv[])
{
   int status = 0;
   PVLayerLoc loc;
   PVPatch ** inPatches, ** outPatches;

   double time = 0.0;
   bool append = false;
   const float minVal = 0.0;
   const float maxVal = 255.0;

   int nx = 64;
   int ny = 64;
   // WARNING - only works for nf==1
   int nf = 1;

   int nxp = 7;
   int nyp = 8;
   int nfp = nf;

   HyPerCol * hc = new HyPerCol("column", argc, argv);

   Communicator * comm = hc->icCommunicator();
   PVParams * params = hc->parameters();

   loc.nx = nx = (int) params->value("column", "nx", (float) nx);
   loc.ny = ny = (int) params->value("column", "ny", (float) ny);

   loc.kx0 = 0 + nx * comm->commColumn();
   loc.ky0 = 0 + ny * comm->commRow();

   loc.nxGlobal = nx * comm->numCommColumns();
   loc.nyGlobal = ny * comm->numCommRows();

   loc.nPad = 0;
   loc.nBands = nf;

   const int numPatches = nx * ny * nf;

   // create weight patches
   //

   inPatches  = init_weight_patches(comm, &loc, nf, nxp, nyp, nfp, false);
   outPatches = init_weight_patches(comm, &loc, nf, nxp, nyp, nfp, true);
   if (inPatches == NULL || outPatches == NULL) {
      status = -1;
      goto finished;
   }

   // write then read/test pvp file
   //

   status = writeWeights(filename, comm, time, append, &loc,
                         nxp, nyp, nfp, minVal, maxVal, inPatches, numPatches);
   if (status != 0) goto finished;

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: wrote weights\n", comm->commRank());
#endif

   status = readWeights(outPatches, numPatches, filename, comm, &time, &loc, true);
   if (status != 0) goto finished;

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: read weights\n", comm->commRank());
#endif

   status = test_output(comm, &loc, nf, inPatches, outPatches);

 finished:

   // TODO - free patches
   delete hc;

   return status;
}

int test_output(Communicator * comm, const PVLayerLoc * loc, int nf,
                PVPatch ** inPatches, PVPatch ** outPatches)
{
   int status = 0;

   const int rank = comm->commRank();
   const int numPatches = loc->nx * loc->ny * nf;

   for (int kl = 0; kl < numPatches; kl++) {
      const int nxp = inPatches[kl]->nx;
      const int nyp = inPatches[kl]->ny;
      const int nfp = inPatches[kl]->nf;
      if (nxp != outPatches[kl]->nx && nyp != outPatches[kl]->ny &&
          nfp != outPatches[kl]->nf &&
          inPatches[kl]->sx != outPatches[kl]->sx &&
          inPatches[kl]->sy != outPatches[kl]->sy &&
          inPatches[kl]->sf != outPatches[kl]->sf)
      {
         status = 2;
         fprintf(stderr, "[%d]: ERROR:test_weights_io: metadata differs at %d\n",
                 rank, kl);
         return status;
      }

      for (int kp = 0; kp < nxp*nyp*nfp; kp++) {
         if (inPatches[kl]->data[kp] != outPatches[kl]->data[kp]) {
            status = 1;
            fprintf(stderr, "[%d]: ERROR:test_weights_io: buffers differ at (%d,%d)"
                    " in==%d out==%d\n", rank, kl, kp,
                    (int) inPatches[kl]->data[kp], (int) outPatches[kl]->data[kp]);
            return status;
         }
      }
   }

   return status;
}

/**
 * Initialize weights with global k+kp indices (mod 256)
 */
PVPatch ** init_weight_patches(Communicator * comm, const PVLayerLoc * loc, int nf,
                               int nxp, int nyp, int nfp, bool zero_flag)
{
   PVPatch ** patches = NULL;

   const int nx  = loc->nx;
   const int ny  = loc->ny;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;

   const int nxGlobal = loc->nxGlobal;
   const int nyGlobal = loc->nyGlobal;

   const int nPatches = nx * ny * nf;

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: kx0==%d ky0==%d\n", comm->commRank(), kx0, ky0);
#endif

   patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   assert(patches != NULL);

   for (int kl = 0; kl < nPatches; kl++) {
      int kx = kx0 + kxPos(kl, nx, ny, nf);
      int ky = ky0 + kyPos(kl, nx, ny, nf);
      int kf = 0;
      int k  = kIndex(kx, ky, kf, nxGlobal, nyGlobal, nf);
      patches[kl] = pvpatch_inplace_new(nxp, nyp, nfp);

      if (zero_flag) {
         for (int kp = 0; kp < nxp*nyp*nfp; kp++) {
            patches[kl]->data[kp] = (unsigned char) 0;
         }
      }
      else {
         for (int kp = 0; kp < nxp*nyp*nfp; kp++) {
            patches[kl]->data[kp] = (unsigned char) (k + kp);
         }
      }
   }

   return patches;
}
