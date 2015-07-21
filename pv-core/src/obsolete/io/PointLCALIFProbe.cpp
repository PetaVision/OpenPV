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

PointLCALIFProbe::PointLCALIFProbe() {
   initPointLCALIFProbe_base();
   // Derived classes should use this PointLCALIFProb constructor, and should call initPointLCALIFProbe during their initialization
}

PointLCALIFProbe::PointLCALIFProbe(const char * probeName, HyPerCol * hc) : PointLIFProbe()
{
   initPointLCALIFProbe_base();
   initPointLCALIFProbe(probeName, hc);
}

PointLCALIFProbe::~PointLCALIFProbe() {
}

int PointLCALIFProbe::initPointLCALIFProbe_base() {
   return PV_SUCCESS;
}

int PointLCALIFProbe::initPointLCALIFProbe(const char * probeName, HyPerCol * hc) {
   return initPointLIFProbe(probeName, hc);
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
int PointLCALIFProbe::writeState(double timed, HyPerLayer * l, int k, int kex)
{
   assert(outputstream && outputstream->fp);
   LCALIFLayer * LCALIF_layer = dynamic_cast<LCALIFLayer *>(l);
   assert(LCALIF_layer != NULL);

   const pvdata_t * V = l->getV();
   const pvdata_t * activity = l->getLayerData();

   if (timed >= writeTime) {
      writeTime += writeStep;
      fprintf(outputstream->fp, "%s t=%.1f k=%d kex=%d", msg, timed, k, kex);
      pvconductance_t * G_E  = LCALIF_layer->getConductance(CHANNEL_EXC);
      pvconductance_t * G_I  = LCALIF_layer->getConductance(CHANNEL_INH);
      pvconductance_t * G_IB = LCALIF_layer->getConductance(CHANNEL_INHB);
      pvgsyndata_t const * gapStrength = LCALIF_layer->getGapStrength();
      pvdata_t * Vth  = LCALIF_layer->getVth();
      const pvdata_t * Vadpt= LCALIF_layer->getVadpt();
      const pvdata_t * integratedSpikeCount = LCALIF_layer->getIntegratedSpikeCount();

      fprintf(outputstream->fp, " G_E=" CONDUCTANCE_PRINT_FORMAT, G_E[k]);
      fprintf(outputstream->fp, " G_I=" CONDUCTANCE_PRINT_FORMAT, G_I[k]);
      fprintf(outputstream->fp, " G_IB=" CONDUCTANCE_PRINT_FORMAT, G_IB[k]);
      if (gapStrength != NULL) fprintf(outputstream->fp, " Gap strength=%6.3f", gapStrength[k]);
      fprintf(outputstream->fp, " integratedSpikeCount=%6.3f", integratedSpikeCount[k]);
      fprintf(outputstream->fp, " Vadpt=%6.3f", Vadpt[k]);
      fprintf(outputstream->fp, " V=%6.3f", V[k]);
      fprintf(outputstream->fp, " Vth=%6.3f", Vth[k]);
      fprintf(outputstream->fp, " Vattained=%6.3f", LCALIF_layer->getVattained()[k]);
      fprintf(outputstream->fp, " Vmeminf=%6.3f", LCALIF_layer->getVmeminf()[k]);
      fprintf(outputstream->fp, " a=%.1f\n", activity[kex]);
      fflush(outputstream->fp);
   }
   return 0;
}


} /* namespace PV */
