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
 * @probeName
 * @hc
 */
StatsProbe::StatsProbe(const char * probeName, HyPerCol * hc) {
   initStatsProbe_base();
   initStatsProbe(probeName, hc);
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
      iotimer->fprint_time(stdout);
      mpitimer->fprint_time(stdout);
      comptimer->fprint_time(stdout);
   }
   delete iotimer;
   delete mpitimer;
   delete comptimer;
}

int StatsProbe::initStatsProbe_base() {
   fMin = FLT_MAX;
   fMax = -FLT_MAX;
   sum = 0.0f;
   sum2 = 0.0f;
   nnz = 0;
   nnzThreshold = (pvdata_t) 0;
   avg = 0.0f;
   sigma = 0.0f;
   type = BufV;
   iotimer = NULL;
   mpitimer = NULL;
   comptimer = NULL;
   return PV_SUCCESS;
}

int StatsProbe::initStatsProbe(const char * probeName, HyPerCol * hc) {
   int status = initLayerProbe(probeName, hc);
   if( status == PV_SUCCESS ) {
      fMin = FLT_MAX;
      fMax = -FLT_MAX;
      sum = 0.0f;
      sum2 = 0.0f;
      avg = 0.0f;
      sigma = 0.0f;
      nnz = 0;
   }
   assert(status == PV_SUCCESS);
   size_t timermessagelen = strlen("StatsProbe ") + strlen(getProbeName()) + strlen(" Comp timer ");
   char timermessage[timermessagelen+1];
   int charsneeded;
   charsneeded = snprintf(timermessage, timermessagelen+1, "StatsProbe %s I/O  timer ", getProbeName());
   assert(charsneeded<=timermessagelen);
   iotimer = new Timer(timermessage);
   charsneeded = snprintf(timermessage, timermessagelen+1, "StatsProbe %s MPI  timer ", getProbeName());
   assert(charsneeded<=timermessagelen);
   mpitimer = new Timer(timermessage);
   charsneeded = snprintf(timermessage, timermessagelen+1, "StatsProbe %s Comp timer ", getProbeName());
   assert(charsneeded<=timermessagelen);
   comptimer = new Timer(timermessage);
   return status;
}

int StatsProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_buffer(ioFlag);
   ioParam_nnzThreshold(ioFlag);
   return status;
}

void StatsProbe::requireType(PVBufType requiredType) {
   PVParams * params = getParentCol()->parameters();
   if (params->stringPresent(getProbeName(), "buffer")) {
      params->handleUnnecessaryStringParameter(getProbeName(), "buffer");
      StatsProbe::ioParam_buffer(PARAMS_IO_READ);
      if (type != requiredType) {
         const char * requiredString = NULL;
         switch (requiredType) {
         case BufV:
            requiredString = "\"MembranePotential\" or \"V\"";
            break;
         case BufActivity:
            requiredString = "\"Activity\" or \"A\"";
            break;
         default:
            assert(0);
            break;
         }
         if (type != BufV) {
            if (getParentCol()->columnId()==0) {
               fprintf(stderr, "   Value \"%s\" is inconsistent with allowed values %s.\n",
                     params->stringValue(getProbeName(), "buffer"), requiredString);
            }
         }
      }
   }
   else {
      type = requiredType;
   }
}

void StatsProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   char * buffer = NULL;
   if (ioFlag == PARAMS_IO_WRITE) {
      switch(type) {
      case BufV:
         buffer = strdup("MembranePotential");
         break;
      case BufActivity:
         buffer = strdup("Activity");
      }
   }
   getParentCol()->ioParamString(ioFlag, getProbeName(), "buffer", &buffer, "Activity", true/*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      assert(buffer);
      size_t len = strlen(buffer);
      for (size_t c=0; c<len; c++) {
         buffer[c] = (char) tolower((int) buffer[c]);
      }
      if (!strcmp(buffer, "v") || !strcmp(buffer, "membranepotential")) {
         type = BufV;
      }
      else if (!strcmp(buffer, "a") || !strcmp(buffer, "activity")) {
         type = BufActivity;
      }
      else {
         if (getParentCol()->columnId()==0) {
            const char * bufnameinparams = getParentCol()->parameters()->stringValue(getProbeName(), "buffer");
            assert(bufnameinparams);
            fprintf(stderr, "%s \"%s\" error: buffer \"%s\" is not recognized.\n",
                  getParentCol()->parameters()->groupKeywordFromName(getProbeName()), getProbeName(), bufnameinparams);
         }
#if PV_USE_MPI
         MPI_Barrier(getParentCol()->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
   }
   free(buffer); buffer = NULL;
}

void StatsProbe::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
    getParentCol()->ioParamValue(ioFlag, getProbeName(), "nnzThreshold", &nnzThreshold, (pvdata_t) 0);
}

/**
 * @time
 * @l
 */
int StatsProbe::outputState(double timed)
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
         fprintf(outputstream->fp, "%sV buffer is NULL\n", getMessage());
         return 0;
      }
      comptimer->start();
      for( int k=0; k<nk; k++ ) {
         pvdata_t a = buf[k];
         sum += a;
         sum2 += a*a;
         if (fabs((double) a)>(double) nnzThreshold) {nnz++;} // Optimize for different datatypes of a?
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
         if (fabs((double) a)>(double) nnzThreshold) {nnz++;} // Optimize for different datatypes of a?
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
      fprintf(outputstream->fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Hz (/dt ms) Max==%f sigma==%f nnz==%i\n", getMessage(), timed,
              nk, (float)sum, fMin, freq, fMax, (float)sigma, nnz);
   }
   else {
      fprintf(outputstream->fp, "%st==%6.1f N==%d Total==%f Min==%f Avg==%f Max==%f sigma==%f nnz==%i\n", getMessage(), timed,
              nk, (float)sum, fMin, (float) avg, fMax, (float) sigma, nnz);
   }

   fflush(outputstream->fp);
   iotimer->stop();

   return PV_SUCCESS;
}

int StatsProbe::checkpointTimers(PV_Stream * timerstream) {
   iotimer->fprint_time(timerstream->fp);
   mpitimer->fprint_time(timerstream->fp);
   comptimer->fprint_time(timerstream->fp);
   return PV_SUCCESS;
}

}
