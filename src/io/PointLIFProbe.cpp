/*
 * PointLIFProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "PointLIFProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../layers/LIF.hpp"
#include <string.h>
#include <iostream>
#include <assert.h>

namespace PV {

/**
 * @filename
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointLIFProbe::PointLIFProbe(const char * filename, int xLoc, int yLoc, int fLoc,
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
PointLIFProbe::PointLIFProbe(int xLoc, int yLoc, int fLoc, const char * msg) :
   LayerProbe()
{
   this->xLoc = xLoc;
   this->yLoc = yLoc;
   this->fLoc = fLoc;
   this->msg = strdup(msg);
   this->sparseOutput = false;
}

PointLIFProbe::~PointLIFProbe()
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
int PointLIFProbe::outputState(float time, HyPerLayer * l)
{
   const PVLayer * clayer = l->clayer;
   LIF * LIF_layer = dynamic_cast<LIF *>(l);
   assert(LIF_layer != NULL);

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;

   const pvdata_t * activity = l->getLayerData();

   const int k = kIndex(xLoc, yLoc, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, clayer->loc.nb);

   if (sparseOutput) {
      fprintf(fp, " (%d %d %3.1f) \n", xLoc, yLoc, activity[kex]);
      // we will control the end of line character from the ConditionalProbe.
   }
   else {
      fprintf(fp, "%s t=%.1f", msg, time);
      pvdata_t * G_E = LIF_layer->getChannel(CHANNEL_EXC);
      pvdata_t * G_I = LIF_layer->getChannel(CHANNEL_INH);
      pvdata_t * G_IB = LIF_layer->getChannel(CHANNEL_INHB);
      pvdata_t * Vth = LIF_layer->getVth();
      fprintf(fp, " G_E=%6.3f", G_E[k]);
      fprintf(fp, " G_I=%6.3f", G_I[k]);
      fprintf(fp, " G_IB=%6.3f", G_IB[k]);
      fprintf(fp, " V=%6.3f", clayer->V[k]);
      fprintf(fp, " Vth=%6.3f", Vth[k]);
      fprintf(fp, " a=%.1f\n", activity[kex]);
      fflush(fp);
   }

   return 0;
}

} // namespace PV
