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
 */
int PointProbe::outputState(float time, HyPerLayer * l)
{
   // LIF * lif = dynamic_cast<LIF*>(l);  // Commented out May 18, 2011.  LIF-specific code moved to PointLIFProbe back in March.

   const PVLayer * clayer = l->clayer;

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;

   const pvdata_t * activity = l->getLayerData();

   const int k = kIndex(xLoc, yLoc, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, clayer->loc.nb);

   if (sparseOutput) {
      fprintf(fp, " (%d %d %3.1f) \n", xLoc, yLoc, activity[kex]);
   }
   else if (activity[kex] != 0.0) {
      fprintf(fp, "%s t=%.1f", msg, time);
      fprintf(fp, " V=%6.5f", clayer->V[k]);
      fprintf(fp, " a=%.5f", activity[kex]);
      fprintf(fp, "\n");
      fflush(fp);
   }

   return 0;
}

} // namespace PV
