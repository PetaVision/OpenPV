/*
 * LCALIFProbe.cpp
 *
 *  Created on: Oct 4, 2012
 *      Author: pschultz
 */

#include "PointLCALIFProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../layers/LCALIFLayer.hpp"
#include <string.h>
#include <assert.h>

namespace PV {

PointLCALIFProbe::PointLCALIFProbe(const char * filename, HyPerLayer * layer, int xLoc, int yLoc, int fLoc,
      const char * msg) : PointLIFProbe(filename, layer, xLoc, yLoc, fLoc, msg)
{
}

PointLCALIFProbe::~PointLCALIFProbe() {
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
int PointLCALIFProbe::writeState(float timef, HyPerLayer * l, int k, int kex)
{
   assert(fp);
   LCALIFLayer * LCALIF_layer = dynamic_cast<LCALIFLayer *>(l);
   assert(LCALIF_layer != NULL);

   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   if (timef >= writeTime) {
      writeTime += writeStep;
      fprintf(fp, "%s t=%.1f k=%d kex=%d", msg, timef, k, kex);
      pvdata_t * G_E  = LCALIF_layer->getConductance(CHANNEL_EXC);
      pvdata_t * G_I  = LCALIF_layer->getConductance(CHANNEL_INH);
      pvdata_t * G_IB = LCALIF_layer->getConductance(CHANNEL_INHB);
      pvdata_t * G_GAP = LCALIF_layer->getConductance(CHANNEL_GAP);
      pvdata_t * Vth  = LCALIF_layer->getVth();
      const pvdata_t * dynVthRest = LCALIF_layer->getDynVthRest();
      const pvdata_t * integratedSpikeCount = LCALIF_layer->getIntegratedSpikeCount();

      fprintf(fp, " G_E=%6.3f", G_E[k]);
      fprintf(fp, " G_I=%6.3f", G_I[k]);
      fprintf(fp, " G_IB=%6.3f", G_IB[k]);
      if (G_GAP != NULL) fprintf(fp, " G_GAP=%6.3f", G_GAP[k]);
      fprintf(fp, " integratedSpikeCount=%6.3f", integratedSpikeCount[k]);
      fprintf(fp, " dynVthRest=%6.3f", dynVthRest[k]);
      fprintf(fp, " V=%6.3f", V[k]);
      fprintf(fp, " Vth=%6.3f", Vth[k]);
      fprintf(fp, " a=%.1f\n", activity[kex]);
      fflush(fp);
   }
   return 0;
}


} /* namespace PV */
