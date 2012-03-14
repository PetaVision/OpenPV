/*
 * ProbeActivityLinear.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: Craig Rasmussen
 */

#include "LinearActivityProbe.hpp"

namespace PV {

/**
 * @layer
 * @dim
 * @kLoc
 * @f
 */
LinearActivityProbe::LinearActivityProbe(HyPerLayer * layer, PVDimType dim, int linePos, int f)
   : LayerProbe()
{
   initLinearActivityProbe(NULL, layer, dim, linePos, f);
}

/**
 * @filename
 * @layer
 * @dim
 * @kLoc
 * @f
 */
LinearActivityProbe::LinearActivityProbe(const char * filename, HyPerLayer * layer, PVDimType dim, int linePos, int f)
    : LayerProbe()
{
   initLinearActivityProbe(filename, layer, dim, linePos, f);
}

LinearActivityProbe::LinearActivityProbe() {
   // Derived classes should call initLinearActivityProbe
}

int LinearActivityProbe::initLinearActivityProbe(const char * filename, HyPerLayer * layer, PVDimType dim, int linePos, int f) {
   initLayerProbe(filename, layer);
   this->hc = layer->getParent();
   this->dim = dim;
   this->linePos = linePos;
   this->f   = f;
   return PV_SUCCESS;
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
int LinearActivityProbe::outputState(float timef)
{
   int width, sLine;
   const float * line;

   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();

   const pvdata_t * activity = getTargetLayer()->getLayerData();

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int nb = loc->nb;

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
   fprintf(fp, "t=%6.1f sum=%3d f=%6.1f Hz :", timef, (int)sum, freq);

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
