/*
 * test_constant_input.cpp
 *
 * This is a full system test with a retina and a spatially constant
 * image.  The retina is non-spiking and connected to layer l1.
 * Using mirror boundary conditions the output of l1 should also be
 * spatially constant.
 *
 *  Created on: Mar 19, 2010
 *      Author: Craig Rasmussen
 */

#include "TestImage.hpp"

#include "../src/columns/HyPerCol.hpp"
#include "../src/connections/HyPerConn.hpp"
#include "../src/layers/V1.hpp"
#include "../src/layers/Retina.hpp"
#include "../src/layers/fileread.h"
#include "../src/io/io.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif

#undef DEBUG_OUTPUT

using namespace PV;

int printLoc(PVLayerLoc loc);
int checkLoc(HyPerCol * hc, PVLayerLoc loc);

int checkInput(PVLayerLoc loc, pvdata_t * data, pvdata_t val, bool extended);

int createTestFile(const char * filename, int count, pvdata_t * buf);
int testOutput(const char * filename, HyPerLayer * l, pvdata_t * inBuf, pvdata_t * outBuf);

const char filename[] = "output/test_layer_direct.bin";
const char outfile[]  = "output/test_layer_direct_out.bin";

int main(int argc, char* argv[])
{
   int status = 0;

   const float imVal = 1.0f;

   // create the managing hypercolumn
   //
   HyPerCol* hc = new HyPerCol("test_constant_input column", argc, argv);

   // create the image
   //
   TestImage * image = new TestImage("test_constant_input image", hc, imVal);

   // create the layers
   //
   HyPerLayer * retina = new Retina("test_constant_input retina", hc, image);
   HyPerLayer * l1     = new V1("test_constant_input layer", hc);

   // create the connections
   //
   new HyPerConn("test_constant_input connection", hc, retina, l1, CHANNEL_EXC);

   hc->initFinish();

   const int rank = hc->columnId();
#ifdef DEBUG_OUTPUT
   printf("[%d]: column: ", rank);
   printLoc(hc->getImageLoc());
   printf("[%d]: image : ", rank);
   printLoc(image->getImageLoc());
   printf("[%d]: retina: ", rank);
   printLoc(*retina->getLayerLoc());
   printf("[%d]: l1    : ", rank);
   printLoc(*l1->getLayerLoc());
#endif

   status = checkLoc(hc, hc->getImageLoc());
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in column loc\n", rank);
      exit(status);
   }

   status = checkLoc(hc, image->getImageLoc());
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in image loc\n", rank);
      exit(status);
   }

   status = checkLoc(hc, *retina->getLayerLoc());
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in retina loc\n", rank);
      exit(status);
   }

   status = checkLoc(hc, *l1->getLayerLoc());
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in l1 loc\n", rank);
      exit(status);
   }

   status = checkInput(image->getImageLoc(), image->getData(), imVal, true);
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in image data\n", rank);
      exit(status);
   }

   pvdata_t * V = retina->clayer->V;
   status = checkInput(*retina->getLayerLoc(), V, imVal, false);
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in retina data\n", rank);
      exit(status);
   }

#ifdef FIXME

   float *inBuf, *outBuf;

   int count = NX*NY;

   inBuf  = (float*) malloc(count*sizeof(float));
   outBuf = (float*) malloc(count*sizeof(float));

    if (rank == 0) {
       //      status = createTestFile(filename, count, inBuf);
       if (status != 0) {
          fprintf(stderr, "[%d]: ERROR - WARNING - exiting without MPI_Finalize()", rank);
          exit(status);
       }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(inBuf);
    free(outBuf);

#endif

    hc->run(2);

   V = l1->clayer->V;
   // there are 4 inputs to each post-synaptic neuron
   int fac = 4;
   status = checkInput(*l1->getLayerLoc(), V, fac*imVal, false);
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in retina data\n", rank);
      exit(status);
   }

   delete hc;

   return status;
}

int checkInput(PVLayerLoc loc, pvdata_t * data, pvdata_t val, bool extended)
{
   int status = 0;

   const int nBorder = (extended) ? loc.nPad : 0;
   const int nk = (loc.nx + 2*nBorder) * (loc.ny + 2*nBorder) * loc.nBands;

   for (int k = 0; k < nk; k++) {
      if (data[k] != val) {
         return -1;
      }
   }

   return status;
}

int testOutput(const char* filename, PV::HyPerLayer* l, float* inBuf, float* outBuf)
{
   int result, err = 0;
   int nTotal = l->clayer->loc.nxGlobal * l->clayer->loc.nyGlobal;
   int rank = l->clayer->columnId;

   FILE* fd = fopen(filename, "rb");
   if (fd == NULL) {
      err = -1;
      fprintf(stderr, "[%d]: ERROR: testOutput: couldn't open file %s\n", rank, filename);
      return err;
   }

    result = fread(outBuf, sizeof(float), nTotal, fd);
    fclose(fd);
    if (result != nTotal) {
       fprintf(stderr, "[ ]: testOutput: ERROR writing to file %s\n", filename);
    }

   for (int i = 0; i < nTotal; i++) {
      if (inBuf[i] != outBuf[i]) {
         err = 1;
         fprintf(stderr, "[%d]: ERROR: testOutput: buffers differ at %d\n", rank, i);
         return err;
      }
   }

   return err;
}


int createTestFile(const char* filename, int nTotal, float* buf)
{
    int i, err = 0;
    size_t result;

    FILE* fd = fopen(filename, "wb");

    for (i = 0; i < nTotal; i++) {
       buf[i] = (float) i;
    }

    result = fwrite(buf, sizeof(float), nTotal, fd);
    fclose(fd);
    if ((int) result != nTotal) {
       fprintf(stderr, "[ ]: createTestFile: ERROR writing to file %s\n", filename);
    }

    fd = fopen(filename, "rb");
    result = fread(buf, sizeof(float), nTotal, fd);
    fclose(fd);
    if ((int) result != nTotal) {
       fprintf(stderr, "[ ]: createTestFile: ERROR reading from file %s\n", filename);
    }

    err = 0;
    for (i = 0; i < nTotal; i++) {
        if (buf[i] != (float) i) {
            err = 1;
            fprintf(stderr, "%s file is incorrect at %d\n", filename, i);
        }
    }

    return err;
}

int printLoc(PVLayerLoc loc)
{
   printf("nxGlobal==%d nyGlobal==%d nx==%d ny==%d kx0==%d ky0==%d nPad==%d nf==%d\n",
     loc.nxGlobal, loc.nyGlobal, loc.nx, loc.ny, loc.kx0, loc.ky0, loc.nPad, loc.nBands);
   fflush(stdout);
   return 0;
}

int checkLoc(HyPerCol * hc, PVLayerLoc loc)
{
   int status = 0;

   const int cols = hc->numCommColumns();
   const int rows = hc->numCommRows();
   const int rank = hc->columnId();

   if (loc.nxGlobal != loc.nx * cols) return -1;
   if (loc.nyGlobal != loc.ny * rows) return -1;

   if (loc.kx0 != loc.nx * hc->commColumn(rank)) return -1;
   if (loc.ky0 != loc.ny * hc->commRow(rank))    return -1;

   return status;
}
