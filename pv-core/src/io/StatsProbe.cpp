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
   int rank = getParent()->columnId();
   if (rank==0) {
      iotimer->fprint_time(stdout);
      mpitimer->fprint_time(stdout);
      comptimer->fprint_time(stdout);
   }
   delete iotimer;
   delete mpitimer;
   delete comptimer;
   free(sum);
   free(sum2);
   free(nnz);
   free(fMin);
   free(fMax);
   free(avg);
   free(sigma);
}

int StatsProbe::initStatsProbe_base() {
   fMin  = NULL;
   fMax  = NULL;
   sum   = NULL;
   sum2  = NULL;
   nnz   = NULL;
   avg   = NULL;
   sigma = NULL;

   type = BufV;
   iotimer = NULL;
   mpitimer = NULL;
   comptimer = NULL;
   nnzThreshold = (pvdata_t) 0;
   return PV_SUCCESS;
}

int StatsProbe::initStatsProbe(const char * probeName, HyPerCol * hc) {
   int status = LayerProbe::initialize(probeName, hc);

   assert(status == PV_SUCCESS);
   size_t timermessagelen = strlen("StatsProbe ") + strlen(getName()) + strlen(" Comp timer ");
   char timermessage[timermessagelen+1];
   int charsneeded;
   charsneeded = snprintf(timermessage, timermessagelen+1, "StatsProbe %s I/O  timer ", getName());
   assert(charsneeded<=timermessagelen);
   iotimer = new Timer(timermessage);
   charsneeded = snprintf(timermessage, timermessagelen+1, "StatsProbe %s MPI  timer ", getName());
   assert(charsneeded<=timermessagelen);
   mpitimer = new Timer(timermessage);
   charsneeded = snprintf(timermessage, timermessagelen+1, "StatsProbe %s Comp timer ", getName());
   assert(charsneeded<=timermessagelen);
   comptimer = new Timer(timermessage);

   int nbatch = hc->getNBatch();

   fMin = (float*) malloc(sizeof(float) * nbatch);
   fMax = (float*) malloc(sizeof(float) * nbatch);
   sum  = (double*) malloc(sizeof(double) * nbatch);
   sum2 = (double*) malloc(sizeof(double) * nbatch);
   avg  = (float*) malloc(sizeof(float) * nbatch);
   sigma = (float*) malloc(sizeof(float) * nbatch);
   nnz  = (int*) malloc(sizeof(int) * nbatch);
   resetStats();

   return status;
}

void StatsProbe::resetStats(){
   for(int b = 0; b < getParent()->getNBatch(); b++){
      fMin[b] = FLT_MAX;
      fMax[b] = -FLT_MAX;
      sum[b] = 0.0f;
      sum2[b] = 0.0f;
      avg[b] = 0.0f;
      sigma[b] = 0.0f;
      nnz[b] = 0;
   }
}

int StatsProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_buffer(ioFlag);
   ioParam_nnzThreshold(ioFlag);
   return status;
}

void StatsProbe::requireType(PVBufType requiredType) {
   PVParams * params = getParent()->parameters();
   if (params->stringPresent(getName(), "buffer")) {
      params->handleUnnecessaryStringParameter(getName(), "buffer");
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
            if (getParent()->columnId()==0) {
               fprintf(stderr, "   Value \"%s\" is inconsistent with allowed values %s.\n",
                     params->stringValue(getName(), "buffer"), requiredString);
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
   getParent()->ioParamString(ioFlag, getName(), "buffer", &buffer, "Activity", true/*warnIfAbsent*/);
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
         if (getParent()->columnId()==0) {
            const char * bufnameinparams = getParent()->parameters()->stringValue(getName(), "buffer");
            assert(bufnameinparams);
            fprintf(stderr, "%s \"%s\" error: buffer \"%s\" is not recognized.\n",
                  this->getKeyword(), getName(), bufnameinparams);
         }
         MPI_Barrier(getParent()->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   free(buffer); buffer = NULL;
}

void StatsProbe::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
    getParent()->ioParamValue(ioFlag, getName(), "nnzThreshold", &nnzThreshold, (pvdata_t) 0);
}

int StatsProbe::initNumValues() {
    return setNumValues(-1);
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
   resetStats();

   nk = getTargetLayer()->getNumNeurons();

   int nbatch = getTargetLayer()->getParent()->getNBatch();

   switch (type) {
   case BufV:
      comptimer->start();
      for(int b = 0; b < nbatch; b++){
         buf = getTargetLayer()->getV() + b * getTargetLayer()->getNumNeurons();
         if( buf == NULL ) {
#ifdef PV_USE_MPI
            if( rank != rcvProc ) return 0;
#endif // PV_USE_MPI
            fprintf(outputstream->fp, "%sV buffer is NULL\n", getMessage());
            return 0;
         }
         for( int k=0; k<nk; k++ ) {
            pvdata_t a = buf[k];
            sum[b] += a;
            sum2[b] += a*a;
            if (fabs((double) a)>(double) nnzThreshold){
               nnz[b]++;
            } // Optimize for different datatypes of a?
            if (a < fMin[b]) fMin[b] = a;
            if (a > fMax[b]) fMax[b] = a;
         }
      }
      comptimer->stop();
      break;
   case BufActivity:
      comptimer->start();
      for(int b = 0; b < nbatch; b++){
         buf = getTargetLayer()->getLayerData() + b * getTargetLayer()->getNumExtended();
         assert(buf != NULL);
         for( int k=0; k<nk; k++ ) {
            const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
            int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up); // TODO: faster to use strides and a double-loop than compute kIndexExtended for every neuron?
            pvdata_t a = buf[kex];
            sum[b] += a;
            sum2[b] += a*a;
            if (fabs((double) a)>(double) nnzThreshold) {
               nnz[b]++; // Optimize for different datatypes of a?
            }
            if( a < fMin[b] ) fMin[b] = a;
            if( a > fMax[b] ) fMax[b] = a;
         }
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

   //In place reduction to prevent allocating a temp recv buffer
   if(rank == rcvProc){
      ierr = MPI_Reduce(MPI_IN_PLACE, sum,  nbatch, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
      ierr = MPI_Reduce(MPI_IN_PLACE, sum2, nbatch, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
      ierr = MPI_Reduce(MPI_IN_PLACE, nnz,  nbatch, MPI_INT, MPI_SUM, rcvProc, comm);
      ierr = MPI_Reduce(MPI_IN_PLACE, fMin,  nbatch, MPI_FLOAT, MPI_MIN, rcvProc, comm);
      ierr = MPI_Reduce(MPI_IN_PLACE, fMax,  nbatch, MPI_FLOAT, MPI_MAX, rcvProc, comm);
      ierr = MPI_Reduce(MPI_IN_PLACE, &nk,   1, MPI_INT, MPI_SUM, rcvProc, comm);
   }
   else{
      ierr = MPI_Reduce(sum, sum, nbatch, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
      ierr = MPI_Reduce(sum2, sum2, nbatch, MPI_DOUBLE, MPI_SUM, rcvProc, comm);
      ierr = MPI_Reduce(nnz, nnz, nbatch, MPI_INT, MPI_SUM, rcvProc, comm);
      ierr = MPI_Reduce(fMin, fMin, nbatch, MPI_FLOAT, MPI_MIN, rcvProc, comm);
      ierr = MPI_Reduce(fMax, fMax, nbatch, MPI_FLOAT, MPI_MAX, rcvProc, comm);
      ierr = MPI_Reduce(&nk, &nk, 1, MPI_INT, MPI_SUM, rcvProc, comm);
      return 0;
   }

   mpitimer->stop();
#endif // PV_USE_MPI
   double divisor = nk;

   iotimer->start();
   for(int b = 0; b < nbatch; b++){
      avg[b] = sum[b]/divisor;
      sigma[b] = sqrt(sum2[b]/divisor - avg[b]*avg[b]);
      if ( type == BufActivity  && getTargetLayer()->getSparseFlag() ) {
         float freq = 1000.0 * avg[b];
         fprintf(outputstream->fp, "%st==%6.1f b==%d N==%d Total==%f Min==%f Avg==%f Hz (/dt ms) Max==%f sigma==%f nnz==%d\n", getMessage(), (double)timed, (int)b, (int)divisor, (double)sum[b], (double)fMin[b], (double)freq, (double)fMax[b], (double)sigma[b], (int)nnz[b]);
      }
      else {
         fprintf(outputstream->fp, "%st==%6.1f b==%d N==%d Total==%f Min==%f Avg==%f Max==%f sigma==%f nnz==%d\n", getMessage(), (double)timed, (int)b, (int)divisor, (double)sum[b], (double)fMin[b], (double) avg[b], (double)fMax[b], (double) sigma[b], (int)nnz[b]);
      }
      fflush(outputstream->fp);
   }

   iotimer->stop();

   return PV_SUCCESS;
}

int StatsProbe::checkpointTimers(PV_Stream * timerstream) {
   iotimer->fprint_time(timerstream->fp);
   mpitimer->fprint_time(timerstream->fp);
   comptimer->fprint_time(timerstream->fp);
   return PV_SUCCESS;
}

BaseObject * createStatsProbe(char const * name, HyPerCol * hc) {
   return hc ? new StatsProbe(name, hc) : NULL;
}

}
