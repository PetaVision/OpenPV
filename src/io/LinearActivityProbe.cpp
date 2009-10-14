/*
 * ProbeActivityLinear.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "LinearActivityProbe.hpp"

namespace PV {

/**
 * @hc
 * @dim
 * @kLoc
 * @f
 */
LinearActivityProbe::LinearActivityProbe(HyPerCol * hc, PVDimType dim, int linePos, int f)
   : PVLayerProbe()
{
   this->parent = hc;
   this->dim = dim;
   this->linePos = linePos;
   this->f   = f;
}

/**
 * @filename
 * @hc
 * @dim
 * @kLoc
 * @f
 */
LinearActivityProbe::LinearActivityProbe(const char * filename, HyPerCol * hc, PVDimType dim, int linePos, int f)
    : PVLayerProbe(filename)
{
   this->parent = hc;
   this->dim = dim;
   this->linePos = linePos;
   this->f   = f;
}

/**
 * @time
 * @l
 */
int LinearActivityProbe::outputState(float time, PVLayer * l)
{
   int width, sLine, k, kex;
   float * line;

   float * activity = l->activity->data;

   const float nx = l->loc.nx;
   const float ny = l->loc.ny;
   const float nf = l->numFeatures;

   const float marginWidth = l->loc.nPad;

   float dt = parent->getDeltaTime();

   if (dim == DimX) {
      width = nx + 2*marginWidth;
      k = kIndex(0, linePos, f, nx, ny, nf);
      sLine = nf;
   }
   else {
      width = ny + 2*marginWidth;
      k = kIndex(linePos, 0, f, nx, ny, nf);
      sLine = nf * (nx + 2*marginWidth);
   }
   kex = kIndexExtended(k, nx, ny, nf, marginWidth) - marginWidth;  // shift so as to display margins
   line = l->activity->data + kex;

   double sum = 0.0;
   for (int k = 0; k < width; k++) {
      float a = line[f + k * sLine];
      sum += a;
   }

   float freq = sum / (width * dt * 0.001);
   fprintf(fp, "t=%6.1f sum=%3d f=%6.1f Hz :", time, (int)sum, freq);

   for (int k = 0; k < width; k++) {
      float a = line[f + k * sLine];
      if (a > 0.0) fprintf(fp, "*");
      else         fprintf(fp, " ");
   }

   fprintf(fp, ":\n");
   fflush(fp);

   return 0;
}

} // namespace PV
