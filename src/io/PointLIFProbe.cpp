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
      const char * msg) : PointProbe(filename, xLoc, yLoc, fLoc, msg)
{
}

/**
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointLIFProbe::PointLIFProbe(int xLoc, int yLoc, int fLoc, const char * msg) :
   PointProbe(xLoc, yLoc, fLoc, msg)
{
}

/**
 * @time
 * @l
 * NOTES:
 *     - Only the activity buffer covers the extended frame - this is the frame that
 * includes boundaries.
 *     - The other dynamic variables (G_E, G_I, V, Vth) cover the "real" or "restricted"
 *     frame.
 *     - When localWmaxFlag is defined in the layer, Wmax covers the extended frame.
 *     - the rate array R is in the restricted space.
 *     - sparseOutput was introduced to deal with ConditionalProbes.
 */
int PointLIFProbe::outputState(float time, HyPerLayer * l)
{
   LIF * LIF_layer = dynamic_cast<LIF *>(l);
   assert(LIF_layer != NULL);

   bool localWmaxFlag = LIF_layer->getLocalWmaxFlag();

   const PVLayerLoc * loc = l->getLayerLoc();

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int nb = loc->nb;

   const pvdata_t * activity = l->getLayerData();

   const int k = kIndex(xLoc, yLoc, fLoc, nx, ny, nf);
   const int kex = kIndexExtended(k, nx, ny, nf, nb);

   if (sparseOutput) {
      fprintf(fp, " (%d %d %3.1f) \n", xLoc, yLoc, activity[kex]);
      // we will control the end of line character from the ConditionalProbe.
   }
   else {
      fprintf(fp, "%s t=%.1f k=%d", msg, time, k);
      pvdata_t * G_E  = LIF_layer->getConductance(CHANNEL_EXC);
      pvdata_t * G_I  = LIF_layer->getConductance(CHANNEL_INH);
      pvdata_t * G_IB = LIF_layer->getConductance(CHANNEL_INHB);
      pvdata_t * Vth  = LIF_layer->getVth();
      fprintf(fp, " G_E=%6.3f", G_E[k]);
      fprintf(fp, " G_I=%6.3f", G_I[k]);
      fprintf(fp, " G_IB=%6.3f", G_IB[k]);
      fprintf(fp, " V=%6.3f", l->clayer->V[k]);
      fprintf(fp, " Vth=%6.3f", Vth[k]);
      // scale rate by 1000.0 to get firing per second
      fprintf(fp, " R=%6.3f", 1000.0 * LIF_layer->getAverageActivity()[k]);
      if (localWmaxFlag){
         fprintf(fp," Wmax=%6.3f",LIF_layer->getWmax()[kex]);
      }
      fprintf(fp, " a=%.1f\n", activity[kex]);
      fflush(fp);
   }

   return 0;
}

} // namespace PV
