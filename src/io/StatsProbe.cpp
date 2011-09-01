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
   fMin = FLT_MAX,
   fMax = -FLT_MAX;
   sum = 0.0f;
   avg = 0.0f;
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
   fMin = FLT_MAX,
   fMax = -FLT_MAX;
   sum = 0.0f;
   avg = 0.0f;

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

#ifdef PV_USE_MPI
   InterColComm * icComm = l->getParent()->icCommunicator();
   MPI_Comm comm = icComm->communicator();
   int ierr;
   const int rcvProc = 0;
   double reducedsum;
   float reducedmin, reducedmax;
   ierr = MPI_Reduce(&sum, &reducedsum, 1, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
   ierr = MPI_Reduce(&fMin, &reducedmin, 1, MPI_FLOAT, MPI_MIN, rcvProc, comm);
   ierr = MPI_Reduce(&fMax, &reducedmax, 1, MPI_FLOAT, MPI_MAX, rcvProc, comm);
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
   sum = reducedsum;
   fMin = reducedmin;
   fMax = reducedmax;
   nk = l->getNumGlobalNeurons();
#endif // PV_USE_MPI
   avg = sum/nk;
   if (type == BufActivity) {
      float freq = 1000.0 * avg;
      fprintf(fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Hz (/dt ms) Max==%f\n", msg, time,
              nk, (float)sum, fMin, freq, fMax);
   }
   else {
      fprintf(fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Max==%f\n", msg, time,
              nk, (float)sum, fMin, (float) avg, fMax);
   }

   fflush(fp);

   return 0;
}

}
