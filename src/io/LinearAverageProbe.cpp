/*
 * LinearAverageProbe.cpp
 *
 *  Created on: Apr 22, 2009
 *      Author: rasmussn
 */

#include "LinearAverageProbe.hpp"
#include "io.h"
#include <assert.h>
#include <string.h>  // strdup

namespace PV {

LinearAverageProbe::LinearAverageProbe() {
   initLinearAverageProbe_base();
   // Derived classes should call initLinearAverageProbe during their own initialization
}

LinearAverageProbe::~LinearAverageProbe()
{
   if (gifFileStream != NULL) {
      PV_fclose(gifFileStream);
   }
   free(gifFilename); gifFilename = NULL;
}

int LinearAverageProbe::initLinearAverageProbe_base() {
   gifFilename = NULL;
   gifFileStream = NULL;
   return PV_SUCCESS;
}

int LinearAverageProbe::initLinearAverageProbe(const char * probeName, HyPerCol * hc) {
   int status = initLinearActivityProbe(probeName, hc);
   this->gifFileStream = NULL;
   return status;
}

int LinearAverageProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LinearActivityProbe::ioParamsFillGroup(ioFlag);
   ioParam_gifFile(ioFlag);
   return status;
}

void LinearAverageProbe::ioParam_gifFile(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamString(ioFlag, getName(), "gifFile", &this->gifFilename, NULL);
}

/**
 * @time
 * @l
 */
int LinearAverageProbe::outputState(double timef)
{
   int nk, sk;
   const pvdata_t * line;

   HyPerLayer * l = getTargetLayer();
   const PVLayer * clayer = l->clayer;

   if (gifFileStream == NULL) {
      int numOnLines = 0;
      char path[PV_PATH_MAX];
      sprintf(path, "%s/%s", l->getParent()->getOutputPath(), gifFilename);

      int nx = clayer->loc.nxGlobal;
      int ny = clayer->loc.nyGlobal;
      int nf = clayer->loc.nf;

      int sx = strideX(&clayer->loc);
      int sy = strideY(&clayer->loc);

      float * buf = (float *) malloc(nx * ny * sizeof(float));
      assert(buf != NULL);

      int err = readFile(path, buf, &nx, &ny);
      if (err == 0) {
         assert(nx <= clayer->loc.nxGlobal);
         assert(ny <= clayer->loc.nyGlobal);
         if (nx < clayer->loc.nxGlobal || ny < clayer->loc.nyGlobal) {
            err = pv_center_image(buf, nx, ny, clayer->loc.nxGlobal, clayer->loc.nyGlobal);
         }
      }

      nx = clayer->loc.nxGlobal;
      ny = clayer->loc.nyGlobal;

      if (dim == DimX) {
         nk = nx;
         sk = nf;
         for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
               if (buf[i*sx + j*sy] > 0.0) {
                  numOnLines += 1;
                  break;
               }
            }
         }

         line = l->getLayerData() + nf * nk * linePos;
      }
      else {
         nk = clayer->activity->loc.ny;
         sk = nf * clayer->activity->loc.nx;
         line = l->getLayerData() + nf * linePos;
      }

      // get list of locations
   }

   float dt = hc->getDeltaTime();
   int nf = clayer->loc.nf;

   if (dim == DimX) {
      nk = clayer->activity->loc.nx;
      sk = nf;
      line = l->getLayerData() + nf * nk * linePos;
   }
   else {
      nk = clayer->activity->loc.ny;
      sk = nf * clayer->activity->loc.nx;
      line = l->getLayerData() + nf * linePos;
   }

   double sum = 0.0;
   for (int k = 0; k < nk; k++) {
      float a = line[f + k * sk];
      sum += a;
   }

   float freq = sum / (nk * dt * 0.001);
   fprintf(outputstream->fp, "t=%4d sum=%3d f=%6.1f Hz :", (int)timef, (int)sum, freq);

   for (int k = 0; k < nk; k++) {
      float a = line[f + k * sk];
      if (a > 0.0) fprintf(outputstream->fp, "*");
      else         fprintf(outputstream->fp, " ");
   }

   fprintf(outputstream->fp, ":\n");
   fflush(outputstream->fp);

   return 0;
}

}
