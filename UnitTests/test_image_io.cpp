/*
 * test_image_io.cpp
 *
 *  Created on: jan 2, 2010
 *      Author: rasmussn
 */

#include "../src/columns/HyPerCol.hpp"
#include "../src/io/imageio.hpp"
#include <stdio.h>

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif

#undef DEBUG_OUTPUT

using namespace PV;

int init_test_buf(Communicator * comm, const PVLayerLoc * loc, unsigned char * buf);
int test_output(Communicator * comm, const PVLayerLoc * loc,
                unsigned char * inBuf, unsigned char * outBuf);

const char file_pvp[] = "output/test_image_io.pvp";
const char file_tif[] = "output/test_image_io.tif";

int main(int argc, char* argv[])
{
   int status = 0;
   PVLayerLoc loc;

   unsigned char * inBuf, * outBuf;

   int nx = 64;
   int ny = 64;
   int nf = 1;

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

   const int localSize = nx * ny * nf;

   inBuf  = (unsigned char *) malloc(localSize * sizeof(unsigned char));
   outBuf = (unsigned char *) malloc(localSize * sizeof(unsigned char));

   // initialize test buffer
   //

   status = init_test_buf(comm, &loc, inBuf);
   if (status != 0) goto finished;

   // write then read/test pvp file
   //

   status = gatherImageFile(file_pvp, comm, &loc, inBuf);
   if (status != 0) goto finished;

   status = scatterImageFile(file_pvp, comm, &loc, outBuf);
   if (status != 0) goto finished;

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: inBuf[0]==%d inBuf[1]==%d outBuf[0]==%d outBuf[1]==%d\n",
           comm->commRank(), (int) inBuf[0], (int) inBuf[1],
           (int) outBuf[0], (int) outBuf[1]);
#endif

   status = test_output(comm, &loc, inBuf, outBuf);

   // write then read/test tif file
   //

   status = gatherImageFile(file_tif, comm, &loc, inBuf);
   if (status != 0) goto finished;

   status = scatterImageFile(file_tif, comm, &loc, outBuf);
   if (status != 0) goto finished;

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: inBuf[0]==%d inBuf[1]==%d outBuf[0]==%d outBuf[1]==%d\n",
           comm->commRank(), (int) inBuf[0], (int) inBuf[1],
           (int) outBuf[0], (int) outBuf[1]);
#endif

   status = test_output(comm, &loc, inBuf, outBuf);

 finished:

   free(inBuf);
   free(outBuf);
   
   delete hc;

   return status;
}

int test_output(Communicator * comm, const PVLayerLoc * loc,
                unsigned char * inBuf, unsigned char * outBuf)
{
   int status = 0;

   const int rank = comm->commRank();
   const int localSize = loc->nx * loc->ny;

   for (int kl = 0; kl < localSize; kl++) {
      if (inBuf[kl] != outBuf[kl]) {
	 status = 1;
	 fprintf(stderr, "[%d]: ERROR:test_image_io: buffers differ at %d"
                 " inBuf==%d outBuf==%d\n", rank, kl, (int) inBuf[kl], outBuf[kl]);
	 return status;
      }
   }

   return status;
}

/**
 * Initialize buffer with global k indices (mod 256)
 */
int init_test_buf(Communicator * comm, const PVLayerLoc * loc, unsigned char * buf)
{
   int status = 0;

   const int nf  = 1;
   const int nx  = loc->nx;
   const int ny  = loc->ny;
   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;

   const int nxGlobal = loc->nxGlobal;
   const int nyGlobal = loc->nyGlobal;

   const int localSize = nx * ny;

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: kx0==%d ky0==%d\n", comm->commRank(), kx0, ky0);
#endif

   for (int kl = 0; kl < localSize; kl++) {
      int kx = kx0 + kxPos(kl, nx, ny, nf);
      int ky = ky0 + kyPos(kl, nx, ny, nf);
      int kf = 0;
      int k  = kIndex(kx, ky, kf, nxGlobal, nyGlobal, nf);
      buf[kl] = (unsigned char) k;
   }

#ifdef DEBUG_OUTPUT
   fprintf(stderr, "[%d]: buf[0]==%d buf[1]==%d\n",
           comm->commRank(), (int) buf[0], (int) buf[1]);
#endif

   return status;
}
