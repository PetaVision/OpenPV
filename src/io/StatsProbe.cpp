/*
 * StatsProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "StatsProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include <float.h>      // FLT_MAX/MIN
#include <string.h>

namespace PV {

/**
 * @filename
 * @type
 * @msg
 */
StatsProbe::StatsProbe(const char * filename, HyPerCol * hc, PVBufType type, const char * msg)
   : LayerProbe(filename, hc)
{
   this->msg = strdup(msg);
   this->type = type;
}

/**
 * @type
 * @msg
 */
StatsProbe::StatsProbe(PVBufType type, const char * msg)
   : LayerProbe()
{
   this->msg = strdup(msg);
   this->type = type;
}

StatsProbe::~StatsProbe()
{
   free(msg);
}

/**
 * @time
 * @l
 */
int StatsProbe::outputState(float time, HyPerLayer * l)
{
   int nk;
   const pvdata_t * buf;
   float fMin = FLT_MAX, fMax = FLT_MIN;
   double sum = 0.0;

   switch (type) {
   case BufV:
      nk  = l->clayer->numNeurons;
      buf = l->clayer->V;
      break;
   case BufActivity:
      nk  = l->clayer->numExtended;
      buf = l->getLayerData();
      break;
   default:
      return 1;
   }

   for (int k = 0; k < nk; k++) {
      pvdata_t a = buf[k];
      sum += a;

      if (a < fMin) fMin = a;
      if (a > fMax) fMax = a;
   }

   if (type == BufActivity) {
      float freq = 1000.0 * (sum/nk);
      fprintf(fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Hz (/dt ms) Max==%f\n", msg, time,
              nk, (float)sum, fMin, freq, fMax);
   }
   else {
      fprintf(fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Max==%f\n", msg, time,
              nk, (float)sum, fMin, (float)(sum / nk), fMax);
   }

   fflush(fp);

   // or just
   // printstats(l);

   return 0;
}

}
