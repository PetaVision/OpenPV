/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "PointProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include <string.h>

namespace PV {

/**
 * @filename
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointProbe::PointProbe(const char * filename, int xLoc, int yLoc, int fLoc,
      const char * msg) :
   LayerProbe(filename)
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
   const PVLayer * clayer = l->clayer;

   const float nx = clayer->loc.nx;
   const float ny = clayer->loc.ny;
   const float nf = clayer->numFeatures;

   const int k = kIndex(xLoc, yLoc, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, clayer->loc.nPad);

   if (sparseOutput) {
      fprintf(fp, " (%d %d %3.1f) \n", xLoc, yLoc, clayer->activity->data[kex]);
      // we will control the end of line character from the ConditionalProbe.
   }
   else {
      fprintf(fp, "%s t=%6.1f", msg, time);
      fprintf(fp, " G_E=%6.3f", clayer->G_E[k]);
      fprintf(fp, " G_I=%6.3f", clayer->G_I[k]);
      fprintf(fp, " V=%6.3f", clayer->V[k]);
      fprintf(fp, " Vth=%6.3f", clayer->Vth[k]);
      fprintf(fp, " a=%3.1f\n", clayer->activity->data[kex]);
      fflush(fp);
   }

   return 0;
}

} // namespace PV
