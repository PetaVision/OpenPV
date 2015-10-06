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

#include <columns/HyPerCol.hpp>
#include <connections/HyPerConn.hpp>
#include <connections/IdentConn.hpp>
#include <layers/Retina.hpp>
#include <io/io.h>

#include <stdio.h>
#include <stdlib.h>

#include <arch/mpi/mpi.h>

#undef DEBUG_OUTPUT

using namespace PV;

int checkLoc(HyPerCol * hc, const PVLayerLoc * loc);

int checkInput(const PVLayerLoc * loc, const pvdata_t * data, pvdata_t val, bool extended);

const char filename[] = "output/test_layer_direct.bin";
const char outfile[]  = "output/test_layer_direct_out.bin";

int main(int argc, char* argv[])
{
   int status = 0;

   PV_Init* initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
   initObj->initialize();

   // create the managing hypercolumn
   //
   HyPerCol* hc = new HyPerCol("test_constant_input column", initObj);

   // create the image
   //
   TestImage * image = new TestImage("test_constant_input image", hc);

   // create the layers
   //
   HyPerLayer * retina = new Retina("test_constant_input retina", hc);

   // create the connections
   //
   HyPerConn * conn = new HyPerConn("test_constant_input connection", hc);
   const int nxp = conn->xPatchSize();
   const int nyp = conn->yPatchSize();
   const PVLayerLoc * imageLoc = image->getLayerLoc();
   const PVLayerLoc * retinaLoc = image->getLayerLoc();
   const int nfPre = imageLoc->nf;
    
   float sumOfWeights = (float) (nxp*nyp*nfPre);
   if (imageLoc->nx > retinaLoc->nx) { sumOfWeights *= imageLoc->nx/retinaLoc->nx;}
   if (imageLoc->ny > retinaLoc->ny) { sumOfWeights *= imageLoc->ny/retinaLoc->ny;}
   
   hc->run();

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

   status = checkLoc(hc, image->getLayerLoc());
   if (status != PV_SUCCESS) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in image loc\n", rank);
      exit(status);
   }

   status = checkLoc(hc, retina->getLayerLoc());
   if (status != PV_SUCCESS) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in retina loc\n", rank);
      exit(status);
   }

   status = checkInput(image->getLayerLoc(), image->getActivity(), image->getConstantVal(), true);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in image data\n", rank);
      exit(status);
   }

   float retinaVal = sumOfWeights * image->getConstantVal();

   status = checkInput(retina->getLayerLoc(), retina->getActivity(), retinaVal, false);
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in retina data\n", rank);
      exit(status);
   }

   status = checkInput(retina->getLayerLoc(), retina->getLayerData(), retinaVal, true);
   if (status != 0) {
      fprintf(stderr, "[%d]: test_constant_input: ERROR in retina data\n", rank);
      exit(status);
   }

   delete hc;
   delete initObj;

   return status;
}

int checkInput(const PVLayerLoc * loc, const pvdata_t * data, pvdata_t val, bool extended)
{
   int status = 0;

   const PVHalo * halo = &loc->halo;
   const int nk = (loc->nx + halo->lt + halo->rt) * (loc->ny + halo->dn + halo->up) * loc->nf;

   for (int k = 0; k < nk; k++) {
      if (data[k] != val) {
         return -1;
      }
   }

   return status;
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

int printLoc(const PVLayerLoc * loc)
{
   printf("nxGlobal==%d nyGlobal==%d nx==%d ny==%d kx0==%d ky0==%d halo==(%d,%d,%d,%d) nf==%d\n",
     loc->nxGlobal, loc->nyGlobal, loc->nx, loc->ny, loc->kx0, loc->ky0, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, loc->nf);
   fflush(stdout);
   return 0;
}

int checkLoc(HyPerCol * hc, const PVLayerLoc * loc)
{
   int status = PV_SUCCESS;

   const int cols = hc->numCommColumns();
   const int rows = hc->numCommRows();
   const int rank = hc->columnId();

   if (loc->nxGlobal != loc->nx * cols) {status = PV_FAILURE;}
   if (loc->nyGlobal != loc->ny * rows) {status = PV_FAILURE;}

   if (loc->kx0 != loc->nx * hc->commColumn()) {status = PV_FAILURE;}
   if (loc->ky0 != loc->ny * hc->commRow())    {status = PV_FAILURE;}

   return status;
}
