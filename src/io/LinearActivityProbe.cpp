/*
 * ProbeActivityLinear.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: Craig Rasmussen
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
   : LayerProbe()
{
   this->hc = hc;
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
    : LayerProbe(filename, hc)
{
   this->hc = hc;
   this->dim = dim;
   this->linePos = linePos;
   this->f   = f;
}

/**
 * @time
 * @l
 * NOTES:
 *    - layer activity lives in an extended space
 *    - by setting dim to PV::dimX or PV::dimY we can plot activity
 *    along the line or along the column.
 *    .
 */
int LinearActivityProbe::outputState(float time, HyPerLayer * l)
{
   int width, sLine;
   const float * line;

   const PVLayer * clayer = l->clayer;

   const pvdata_t * activity = l->getLayerData();

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const int nb = clayer->loc.nb;

   float dt = hc->getDeltaTime();

   double sum = 0.0;
   float freq;

   if (dim == DimX) {
      width = nx + 2*nb;
      line = activity + (linePos + nb) * width * nf;
      sLine = nf;
   }
   else {
      width = ny + 2*nb;
      line = activity + (linePos + nb)*nf;
      sLine = nf * (nx + 2 * nb);

   }

   for (int k = 0; k < width; k++) {
     float a = line[f + k * sLine];
     sum += a;
   }

   freq = sum / (width * dt * 0.001);
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
