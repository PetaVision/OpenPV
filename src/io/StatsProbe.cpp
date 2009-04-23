/*
 * StatsProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "StatsProbe.hpp"
#include <float.h>      // FLT_MAX/MIN
#include <string.h>

namespace PV {

StatsProbe::StatsProbe(const char * filename, PVBufType type, const char * msg)
   : PVLayerProbe(filename)
{
   this->msg = strdup(msg);
   this->type = type;
}

StatsProbe::StatsProbe(PVBufType type, const char * msg)
   : PVLayerProbe()
{
   this->msg = strdup(msg);
   this->type = type;
}

StatsProbe::~StatsProbe()
{
   free(msg);
}

int StatsProbe::outputState(float time, PVLayer * l)
{
   pvdata_t * buf;
   float fMin = FLT_MAX, fMax = FLT_MIN;
   double sum = 0.0;

   switch (type) {
   case BufV:         buf = l->V;               break;
   case BufActivity:  buf = l->activity->data;  break;
   default:
      return 1;
   }

   int nk = l->numNeurons;

   for (int k = 0; k < nk; k++) {
      pvdata_t a = buf[k];
      sum += a;

      if (a < fMin) fMin = a;
      if (a > fMax) fMax = a;
   }

   fprintf(fp, "%s t=%4d N=%d Total=%f Min=%f, Avg=%f, Max=%f\n", msg, (int)time,
           nk, (float)sum, fMin, (float)(sum / nk), fMax);
   fflush(fp);

   // or just
   // printstats(l);

   return 0;
}

}
