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
 * @hc
 * @msg
 */
StatsProbe::StatsProbe(const char * filename, HyPerLayer * layer, const char * msg)
   : LayerProbe()
{
   initStatsProbe(filename, layer, BufActivity, msg);
}

/**
 * @msg
 */
StatsProbe::StatsProbe(HyPerLayer * layer, const char * msg)
   : LayerProbe()
{
   initStatsProbe(NULL, layer, BufActivity, msg);
}

/**
 * @filename
 * @hc
 * @type
 * @msg
 */
StatsProbe::StatsProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg)
   : LayerProbe()
{
   initStatsProbe(filename, layer, type, msg);
}

/**
 * @type
 * @msg
 */
StatsProbe::StatsProbe(HyPerLayer * layer, PVBufType type, const char * msg)
   : LayerProbe()
{
   initStatsProbe(NULL, layer, type, msg);
}

StatsProbe::StatsProbe()
   : LayerProbe()
{
   // Derived classes should call initStatsProbe
}

StatsProbe::~StatsProbe()
{
   free(msg);
}

int StatsProbe::initStatsProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg) {
   int status = initLayerProbe(filename, layer);
   if( status == PV_SUCCESS ) {
      fMin = FLT_MAX;
      fMax = -FLT_MAX;
      sum = 0.0f;
      avg = 0.0f;
      this->type = type;
      status = initMessage(msg);
   }
   assert(status == PV_SUCCESS);
   return status;
}

int StatsProbe::initMessage(const char * msg) {
   int status = PV_SUCCESS;
   if( msg != NULL && msg[0] != '\0' ) {
      size_t msglen = strlen(msg);
      this->msg = (char *) calloc(msglen+2, sizeof(char)); // Allocate room for colon plus null terminator
      if(this->msg) {
         memcpy(this->msg, msg, msglen);
         this->msg[msglen] = ':';
         this->msg[msglen+1] = '\0';
      }
   }
   else {
      this->msg = (char *) calloc(1, sizeof(char));
      if(this->msg) {
         this->msg[0] = '\0';
      }
   }
   if( !this->msg ) {
      fprintf(stderr, "StatsProbe: Unable to allocate memory for probe's message.\n");
      status = PV_FAILURE;
   }
   assert(status == PV_SUCCESS);
   return status;
}

/**
 * @time
 * @l
 */
int StatsProbe::outputState(float timef)
{
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   MPI_Comm comm = icComm->communicator();
   int rank = icComm->commRank();
   const int rcvProc = 0;
#endif // PV_USE_MPI

   int nk;
   const pvdata_t * buf;
   fMin = FLT_MAX,
   fMax = -FLT_MAX;
   sum = 0.0f;
   avg = 0.0f;

   nk = getTargetLayer()->getNumNeurons();
   switch (type) {
   case BufV:
      buf = getTargetLayer()->getV();
      if( buf == NULL ) {
#ifdef PV_USE_MPI
         if( rank != rcvProc ) return 0;
#endif // PV_USE_MPI
         fprintf(fp, "%sV buffer is NULL\n", msg);
         return 0;
      }
      for( int k=0; k<nk; k++ ) {
         pvdata_t a = buf[k];
         sum += a;
         if (a < fMin) fMin = a;
         if (a > fMax) fMax = a;
      }
      break;
   case BufActivity:
      buf = getTargetLayer()->getLayerData();
      assert(buf != NULL);
      for( int k=0; k<nk; k++ ) {
         const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
         int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
         pvdata_t a = buf[kex];
         sum += a;
         if( a < fMin ) fMin = a;
         if( a > fMax ) fMax = a;
      }
      break;
   default:
      assert(0);
      break;
   }

#ifdef PV_USE_MPI
   int ierr;
   double reducedsum;
   float reducedmin, reducedmax;
   int totalNeurons;
   ierr = MPI_Reduce(&sum, &reducedsum, 1, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
   ierr = MPI_Reduce(&fMin, &reducedmin, 1, MPI_FLOAT, MPI_MIN, rcvProc, comm);
   ierr = MPI_Reduce(&fMax, &reducedmax, 1, MPI_FLOAT, MPI_MAX, rcvProc, comm);
   ierr = MPI_Reduce(&nk, &totalNeurons, 1, MPI_INT, MPI_SUM, rcvProc, comm);
   if( rank != rcvProc ) {
      return 0;
   }
   sum = reducedsum;
   fMin = reducedmin;
   fMax = reducedmax;
   nk = totalNeurons;
#endif // PV_USE_MPI
   avg = sum/nk;
   if ( type == BufActivity  && getTargetLayer()->getSpikingFlag() ) {
      float freq = 1000.0 * avg;
      fprintf(fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Hz (/dt ms) Max==%f\n", msg, timef,
              nk, (float)sum, fMin, freq, fMax);
   }
   else {
      fprintf(fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Max==%f\n", msg, timef,
              nk, (float)sum, fMin, (float) avg, fMax);
   }

   fflush(fp);

   return 0;
}

}
