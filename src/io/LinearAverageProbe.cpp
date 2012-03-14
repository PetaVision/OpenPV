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

/**
 * @hc
 * @dim
 * @f
 * @gifFile
 */
LinearAverageProbe::LinearAverageProbe(HyPerLayer * layer, PVDimType dim, int f, const char * gifFile)
   : LinearActivityProbe()
{
   initLinearAverageProbe(NULL, layer, dim, f, gifFile);
}

/**
 * @filename
 * @hc
 * @dim
 * @f
 * @char
 */
LinearAverageProbe::LinearAverageProbe(const char * filename, HyPerLayer * layer, PVDimType dim, int f, const char * gifFile)
    : LinearActivityProbe()
{
   initLinearAverageProbe(filename, layer, dim, f, gifFile);
}

LinearAverageProbe::~LinearAverageProbe()
{
   if (fpGif != NULL) {
      fclose(fpGif);
   }
}

int LinearAverageProbe::initLinearAverageProbe(const char * filename, HyPerLayer * layer, PVDimType dim, int f, const char * gifFile) {

   initLinearActivityProbe(filename, layer, dim, 0, f);
   this->gifFile = strdup(gifFile);
   this->fpGif   = NULL;
   return PV_SUCCESS;
}

/**
 * @time
 * @l
 */
int LinearAverageProbe::outputState(float timef, HyPerLayer * l)
{
   int nk, sk;
   const pvdata_t * line;

   const PVLayer * clayer = l->clayer;

   if (fpGif == NULL) {
      int numOnLines = 0;
      char path[PV_PATH_MAX];
      sprintf(path, "%s/%s", l->parent->getOutputPath(), gifFile);
//      fpGif = fopen(path, "r");

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
   fprintf(fp, "t=%4d sum=%3d f=%6.1f Hz :", (int)timef, (int)sum, freq);

   for (int k = 0; k < nk; k++) {
      float a = line[f + k * sk];
      if (a > 0.0) fprintf(fp, "*");
      else         fprintf(fp, " ");
   }

   fprintf(fp, ":\n");
   fflush(fp);

   return 0;
}

}
