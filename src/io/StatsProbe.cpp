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
   initStatsProbe_base();
   initStatsProbe(filename, layer, BufActivity, msg);
}

/**
 * @msg
 */
StatsProbe::StatsProbe(HyPerLayer * layer, const char * msg)
   : LayerProbe()
{
   initStatsProbe_base();
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
   initStatsProbe_base();
   initStatsProbe(filename, layer, type, msg);
}

/**
 * @type
 * @msg
 */
StatsProbe::StatsProbe(HyPerLayer * layer, PVBufType type, const char * msg)
   : LayerProbe()
{
   initStatsProbe_base();
   initStatsProbe(NULL, layer, type, msg);
}

StatsProbe::StatsProbe()
   : LayerProbe()
{
   initStatsProbe_base();
   // Derived classes should call initStatsProbe
}

StatsProbe::~StatsProbe()
{
   int rank = getTargetLayer()->getParent()->columnId();
   if (rank==0) {
      printf("StatsProbe %s I/O  timer ", msg); // Lack of \n is deliberate, elapsed_time() calls printf with \n.
      iotimer->elapsed_time();
      printf("StatsProbe %s MPI  timer ", msg);
      mpitimer->elapsed_time();
      printf("StatsProbe %s Comp timer ", msg);
      comptimer->elapsed_time();
   }
   delete iotimer;
   delete mpitimer;
   delete comptimer;
   free(msg);
}

int StatsProbe::initStatsProbe_base() {
   fMin = FLT_MAX;
   fMax = -FLT_MAX;
   sum = 0.0f;
   sum2 = 0.0f;
   nnz = 0;
   avg = 0.0f;
   sigma = 0.0f;
   type = BufV;
   msg = NULL;
   iotimer = NULL;
   mpitimer = NULL;
   comptimer = NULL;
   return PV_SUCCESS;
}

int StatsProbe::initStatsProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg) {
   int status = initLayerProbe(filename, layer);
   if( status == PV_SUCCESS ) {
      fMin = FLT_MAX;
      fMax = -FLT_MAX;
      sum = 0.0f;
      sum2 = 0.0f;
      avg = 0.0f;
      sigma = 0.0f;
      nnz = 0;
      this->type = type;
      status = initMessage(msg);
   }
   assert(status == PV_SUCCESS);
   iotimer = new Timer();
   mpitimer = new Timer();
   comptimer = new Timer();
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
int StatsProbe::outputState(double timef)
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
   sum2 = 0.0f;
   avg = 0.0f;
   sigma = 0.0f;
   nnz = 0;

   nk = getTargetLayer()->getNumNeurons();
   switch (type) {
   case BufV:
      buf = getTargetLayer()->getV();
      if( buf == NULL ) {
#ifdef PV_USE_MPI
         if( rank != rcvProc ) return 0;
#endif // PV_USE_MPI
         fprintf(outputstream->fp, "%sV buffer is NULL\n", msg);
         return 0;
      }
      comptimer->start();
      for( int k=0; k<nk; k++ ) {
         pvdata_t a = buf[k];
         sum += a;
         sum2 += a*a;
         nnz += (int) (a>0);
         if (a < fMin) fMin = a;
         if (a > fMax) fMax = a;
      }
      comptimer->stop();
      break;
   case BufActivity:
      comptimer->start();
      buf = getTargetLayer()->getLayerData();
      assert(buf != NULL);
      for( int k=0; k<nk; k++ ) {
         const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
         int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb); // TODO: faster to use strides and a double-loop than compute kIndexExtended for every neuron?
         pvdata_t a = buf[kex];
         sum += a;
         sum2 += a*a;
         nnz += (int) (a>0);
         if( a < fMin ) fMin = a;
         if( a > fMax ) fMax = a;
      }
      comptimer->stop();
      break;
   default:
      assert(0);
      break;
   }

#ifdef PV_USE_MPI
   mpitimer->start();
   int ierr;
   double reducedsum, reducedsum2;
   int reducednnz;
   float reducedmin, reducedmax;
   int totalNeurons;
   ierr = MPI_Reduce(&sum, &reducedsum, 1, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
   ierr = MPI_Reduce(&sum2, &reducedsum2, 1, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
   ierr = MPI_Reduce(&nnz, &reducednnz, 1, MPI_INT, MPI_SUM, rcvProc, comm);
   ierr = MPI_Reduce(&fMin, &reducedmin, 1, MPI_FLOAT, MPI_MIN, rcvProc, comm);
   ierr = MPI_Reduce(&fMax, &reducedmax, 1, MPI_FLOAT, MPI_MAX, rcvProc, comm);
   ierr = MPI_Reduce(&nk, &totalNeurons, 1, MPI_INT, MPI_SUM, rcvProc, comm);
   if( rank != rcvProc ) {
      return 0;
   }
   sum = reducedsum;
   sum2 = reducedsum2;
   nnz = reducednnz;
   fMin = reducedmin;
   fMax = reducedmax;
   nk = totalNeurons;
   mpitimer->stop();
#endif // PV_USE_MPI
   iotimer->start();
   avg = sum/nk;
   sigma = sqrt(sum2/nk - avg*avg);
   if ( type == BufActivity  && getTargetLayer()->getSpikingFlag() ) {
      float freq = 1000.0 * avg;
      fprintf(outputstream->fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Hz (/dt ms) Max==%f sigma==%f nnz==%i\n", msg, timef,
              nk, (float)sum, fMin, freq, fMax, (float)sigma, nnz);
   }
   else {
      fprintf(outputstream->fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Max==%f sigma==%f nnz==%i\n", msg, timef,
              nk, (float)sum, fMin, (float) avg, fMax, (float) sigma, nnz);
   }

   fflush(outputstream->fp);
   iotimer->stop();

   return 0;
}

}
