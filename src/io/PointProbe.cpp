/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "PointProbe.hpp"
#include "../layers/HyPerLayer.hpp"
// #include "../layers/LIF.hpp" // Commented out May 18, 2011.  LIF-specific code moved to PointLIFProbe back in March.
#include <string.h>

namespace PV {

/**
 * @filename
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(const char * filename, HyPerCol * hc, int xLoc, int yLoc, int fLoc,
      const char * msg) :
   LayerProbe(filename, hc)
{
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   this->msg = strdup(msg);
   this->sparseOutput = false;
}

/**
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(int xLoc, int yLoc, int fLoc, const char * msg) :
   LayerProbe()
{
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
int PointProbe::outputState(float time, HyPerLayer * l)
{
   // LIF * lif = dynamic_cast<LIF*>(l);  // Commented out May 18, 2011.  LIF-specific code moved to PointLIFProbe back in March.

   const PVLayerLoc * loc = l->getLayerLoc();

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

   return writeState(time, l, k, kex);
}

/**
 * @time
 * @l
 * @k
 * @kex
 */
int PointProbe::writeState(float time, HyPerLayer * l, int k, int kex) {

   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   if (sparseOutput) {
      fprintf(fp, " (%d %d %3.1f) \n", xLoc, yLoc, activity[kex]);
   }
   else if (activity[kex] != 0.0) {
      fprintf(fp, "%s t=%.1f", msg, time);
      fprintf(fp, " V=%6.5f", V[k]);
      fprintf(fp, " a=%.5f", activity[kex]);
      fprintf(fp, "\n");
      fflush(fp);
   }

   return PV_SUCCESS;
}

} // namespace PV
