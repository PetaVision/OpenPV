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

LinearAverageProbe::LinearAverageProbe(HyPerCol * hc, PVDimType dim, int f, const char * gifFile)
   : LinearActivityProbe(hc, dim, 0, f)
{
   this->gifFile = strdup(gifFile);
   this->fpGif   = NULL;
}

LinearAverageProbe::LinearAverageProbe(const char * filename, HyPerCol * hc, PVDimType dim, int f, const char * gifFile)
    : LinearActivityProbe(filename, hc, dim, 0, f)
{
   this->gifFile = strdup(gifFile);
   this->fpGif   = NULL;
}

LinearAverageProbe::~LinearAverageProbe()
{
   if (fpGif != NULL) {
      fclose(fpGif);
   }
}

int LinearAverageProbe::outputState(float time, PVLayer * l)
{
   int nk, sk;
   float * line;

   if (fpGif == NULL) {
      int numOnLines = 0;
      char path[PV_PATH_MAX];
      sprintf(path, "%s%s", OUTPUT_PATH, gifFile);
//      fpGif = fopen(path, "r");

      int nx = l->loc.nxGlobal;
      int ny = l->loc.nyGlobal;
      int nf = l->numFeatures;

      int sx = strideX(nx, ny, nf);
      int sy = strideY(nx, ny, nf);
      int sf = strideF(nx, ny, nf);

      float * buf = (float *) malloc(nx * ny * sizeof(float));
      assert(buf != NULL);

      int err = readFile(path, buf, &nx, &ny);
      if (err == 0) {
         assert(nx <= l->loc.nxGlobal);
         assert(ny <= l->loc.nyGlobal);
         if (nx < l->loc.nxGlobal || ny < l->loc.nyGlobal) {
            err = pv_center_image(buf, nx, ny, l->loc.nxGlobal, l->loc.nyGlobal);
         }
      }

      nx = l->loc.nxGlobal;
      ny = l->loc.nyGlobal;

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

         line = l->activity->data + nf * nk * loc;
      }
      else {
         nk = l->activity->loc.ny;
         sk = nf * l->activity->loc.nx;
         line = l->activity->data + nf * loc;
      }

      // get list of locations
   }

   float dt = parent->getDeltaTime();
   int nf = l->numFeatures;

   if (dim == DimX) {
      nk = l->activity->loc.nx;
      sk = nf;
      line = l->activity->data + nf * nk * loc;
   }
   else {
      nk = l->activity->loc.ny;
      sk = nf * l->activity->loc.nx;
      line = l->activity->data + nf * loc;
   }

   double sum = 0.0;
   for (int k = 0; k < nk; k++) {
      float a = line[f + k * sk];
      sum += a;
   }

   float freq = sum / (nk * dt * 0.001);
   fprintf(fp, "t=%4d sum=%3d f=%6.1f Hz :", (int)time, (int)sum, freq);

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
