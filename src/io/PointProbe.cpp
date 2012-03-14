/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "PointProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include <string.h>

namespace PV {

/**
 * @filename
 * @layer
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc,
      const char * msg) :
   LayerProbe()
{
   initLayerProbe(filename, layer);
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   this->msg = strdup(msg);
   this->sparseOutput = false;
}

/**
 * @layer
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) :
   LayerProbe()
{
   initLayerProbe(NULL, layer);
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   this->msg = strdup(msg);
   this->sparseOutput = false;
}

PointProbe::~PointProbe()
{
   free(msg);
}

/**
 * @time
 * @l
 * NOTES:
 *     - Only the activity buffer covers the extended frame - this is the frame that
 * includes boundaries.
 *     - The other dynamic variables (G_E, G_I, V, Vth) cover the "real" or "restricted"
 *     frame.
 *     - sparseOutput was introduced to deal with ConditionalProbes.
 *     - In MPI runs, xLoc and yLoc refer to global coordinates.
 *     writeState is only called by the processor with (xLoc,yLoc) in its
 *     non-extended region.
 */
int PointProbe::outputState(float timef)
{
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();

   const int kx0 = loc->kx0;
   const int ky0 = loc->ky0;
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int xLocLocal = xLoc - kx0;
   const int yLocLocal = yLoc - ky0;
   if( xLocLocal < 0 || xLocLocal >= nx ||
       yLocLocal < 0 || yLocLocal >= ny) return PV_SUCCESS;
   const int nf = loc->nf;

   const int k = kIndex(xLocLocal, yLocLocal, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, loc->nb);

   return writeState(timef, getTargetLayer(), k, kex);
}

/**
 * @time
 * @l
 * @k
 * @kex
 */
int PointProbe::writeState(float timef, HyPerLayer * l, int k, int kex) {

   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   if (sparseOutput) {
      fprintf(fp, " (%d %d %3.1f) \n", xLoc, yLoc, activity[kex]);
   }
   else if (activity[kex] != 0.0) {
      fprintf(fp, "%s t=%.1f", msg, timef);
      fprintf(fp, " V=%6.5f", V != NULL ? V[k] : 0.0f);
      fprintf(fp, " a=%.5f", activity[kex]);
      fprintf(fp, "\n");
      fflush(fp);
   }

   return PV_SUCCESS;
}

} // namespace PV
