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
   int width, sLine;
   float * line;
   const LayerLoc * loc = &l->activity->loc;

   float dt = parent->getDeltaTime();
   int nf = l->numFeatures;

   if (dim == DimX) {
      width = loc->nx + 2*loc->nPad;
      sLine = nf;
      line  = l->activity->data + linePos * width * nf;
   }
   else {
      width = loc->ny + 2*loc->nPad;
      sLine = nf * (loc->nx + 2*loc->nPad);
      line  = l->activity->data + linePos * nf;
   }

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
