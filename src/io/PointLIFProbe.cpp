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
 * @layer
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointLIFProbe::PointLIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc,
      const char * msg) : PointProbe(filename, layer, xLoc, yLoc, fLoc, msg)
{
   writeTime = 0.0;
   writeStep = layer->getParent()->getDeltaTime();  // Marian, don't change this default behavior
}

/**
 * @xLoc
 * @yLoc
 * @fLoc
 * @msg
 */
PointLIFProbe::PointLIFProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, const char * msg) :
   PointProbe(layer, xLoc, yLoc, fLoc, msg)
{
   writeTime = 0.0;
   writeStep = 10.0;
}

PointLIFProbe::PointLIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc,
      float writeStep, const char * msg) : PointProbe(filename, layer, xLoc, yLoc, fLoc, msg)
{
   writeTime = 0.0;
   this->writeStep = writeStep;
}

PointLIFProbe::PointLIFProbe(HyPerLayer * layer, int xLoc, int yLoc, int fLoc, float writeStep, const char * msg) :
   PointProbe(layer, xLoc, yLoc, fLoc, msg)
{
   writeTime = 0.0;
   this->writeStep = writeStep;
}

/**
 * @time
 * @l
 * @k
 * @kex
 * NOTES:
 *     - Only the activity buffer covers the extended frame - this is the frame that
 * includes boundaries.
 *     - The other dynamic variables (G_E, G_I, V, Vth) cover the "real" or "restricted"
 *     frame.
 *     - sparseOutput was introduced to deal with ConditionalProbes.
 */
int PointLIFProbe::writeState(double timed, HyPerLayer * l, int k, int kex)
{
   assert(fp);
   LIF * LIF_layer = dynamic_cast<LIF *>(l);
   assert(LIF_layer != NULL);

   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   if (timed >= writeTime) {
      writeTime += writeStep;
      fprintf(fp, "%s t=%.1f k=%d", msg, timed, k);
      pvdata_t * G_E  = LIF_layer->getConductance(CHANNEL_EXC);
      pvdata_t * G_I  = LIF_layer->getConductance(CHANNEL_INH);
      pvdata_t * G_IB = LIF_layer->getConductance(CHANNEL_INHB);
      pvdata_t * G_GAP = LIF_layer->getConductance(CHANNEL_GAP);
      pvdata_t * Vth  = LIF_layer->getVth();

      fprintf(fp, " G_E=%6.3f", G_E[k]);
      fprintf(fp, " G_I=%6.3f", G_I[k]);
      fprintf(fp, " G_IB=%6.3f", G_IB[k]);
      if (G_GAP != NULL) fprintf(fp, " G_GAP=%6.3f", G_GAP[k]);
      fprintf(fp, " V=%6.3f", V[k]);
      fprintf(fp, " Vth=%6.3f", Vth[k]);
      fprintf(fp, " a=%.1f\n", activity[kex]);
      fflush(fp);
   }
   return 0;
}

} // namespace PV
