/*
 * StatsProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "StatsProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include <float.h> // FLT_MAX/MIN
#include <string.h>

namespace PV {

StatsProbe::StatsProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

StatsProbe::StatsProbe() : LayerProbe() {
   initialize_base();
   // Derived classes should call initialize
}

StatsProbe::~StatsProbe() {
   int rank = mCommunicator->commRank();
   if (rank == 0 and !mOutputStreams.empty()) {
      iotimer->fprint_time(output(0));
      mpitimer->fprint_time(output(0));
      comptimer->fprint_time(output(0));
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

int StatsProbe::initialize_base() {
   fMin  = NULL;
   fMax  = NULL;
   sum   = NULL;
   sum2  = NULL;
   nnz   = NULL;
   avg   = NULL;
   sigma = NULL;

   type         = BufV;
   iotimer      = NULL;
   mpitimer     = NULL;
   comptimer    = NULL;
   nnzThreshold = (float)0;
   return PV_SUCCESS;
}

void StatsProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   LayerProbe::initialize(name, params, comm);
}

void StatsProbe::resetStats() {
   for (int b = 0; b < mLocalBatchWidth; b++) {
      fMin[b]  = FLT_MAX;
      fMax[b]  = -FLT_MAX;
      sum[b]   = 0.0f;
      sum2[b]  = 0.0f;
      avg[b]   = 0.0f;
      sigma[b] = 0.0f;
      nnz[b]   = 0;
   }
}

int StatsProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_buffer(ioFlag);
   ioParam_nnzThreshold(ioFlag);
   return status;
}

void StatsProbe::requireType(PVBufType requiredType) {
   PVParams *params = parameters();
   if (params->stringPresent(getName(), "buffer")) {
      params->handleUnnecessaryStringParameter(getName(), "buffer");
      StatsProbe::ioParam_buffer(PARAMS_IO_READ);
      if (type != requiredType) {
         const char *requiredString = NULL;
         switch (requiredType) {
            case BufV: requiredString        = "\"MembranePotential\" or \"V\""; break;
            case BufActivity: requiredString = "\"Activity\" or \"A\""; break;
            default: assert(0); break;
         }
         if (type != BufV) {
            if (mCommunicator->globalCommRank() == 0) {
               ErrorLog().printf(
                     "   Value \"%s\" is inconsistent with allowed values %s.\n",
                     params->stringValue(getName(), "buffer"),
                     requiredString);
            }
         }
      }
   }
   else {
      type = requiredType;
   }
}

void StatsProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   char *buffer = NULL;
   if (ioFlag == PARAMS_IO_WRITE) {
      switch (type) {
         case BufV: buffer        = strdup("MembranePotential"); break;
         case BufActivity: buffer = strdup("Activity");
      }
   }
   parameters()->ioParamString(
         ioFlag, getName(), "buffer", &buffer, "Activity", true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      assert(buffer);
      size_t len = strlen(buffer);
      for (size_t c = 0; c < len; c++) {
         buffer[c] = (char)tolower((int)buffer[c]);
      }
      if (!strcmp(buffer, "v") || !strcmp(buffer, "membranepotential")) {
         type = BufV;
      }
      else if (!strcmp(buffer, "a") || !strcmp(buffer, "activity")) {
         type = BufActivity;
      }
      else {
         if (mCommunicator->commRank() == 0) {
            const char *bufnameinparams = parameters()->stringValue(getName(), "buffer");
            assert(bufnameinparams);
            ErrorLog().printf(
                  "%s: buffer \"%s\" is not recognized.\n", getDescription_c(), bufnameinparams);
         }
         MPI_Barrier(mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
   }
   free(buffer);
   buffer = NULL;
}

void StatsProbe::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "nnzThreshold", &nnzThreshold, 0.0f);
}

void StatsProbe::initNumValues() { setNumValues(-1); }

Response::Status StatsProbe::allocateDataStructures() {
   auto status = LayerProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   int const nbatch = mLocalBatchWidth;

   fMin  = (float *)malloc(sizeof(float) * nbatch);
   fMax  = (float *)malloc(sizeof(float) * nbatch);
   sum   = (double *)malloc(sizeof(double) * nbatch);
   sum2  = (double *)malloc(sizeof(double) * nbatch);
   avg   = (float *)malloc(sizeof(float) * nbatch);
   sigma = (float *)malloc(sizeof(float) * nbatch);
   nnz   = (int *)malloc(sizeof(int) * nbatch);
   resetStats();

   return Response::SUCCESS;
}

Response::Status
StatsProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = LayerProbe::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;

   std::string timermessagehead;
   timermessagehead.append("StatsProbe ").append(getName());
   std::string timermessage;

   timermessage = timermessagehead + " I/O  timer ";
   iotimer      = new Timer(timermessage.c_str());
   checkpointer->registerTimer(iotimer);

   timermessage = timermessagehead + " MPI  timer ";
   mpitimer     = new Timer(timermessage.c_str());
   checkpointer->registerTimer(mpitimer);

   timermessage = timermessagehead + " Comp timer ";
   comptimer    = new Timer(timermessage.c_str());
   checkpointer->registerTimer(comptimer);
   return Response::SUCCESS;
}

Response::Status StatsProbe::outputState(double simTime, double deltaTime) {
#ifdef PV_USE_MPI
   Communicator const *icComm = mCommunicator;
   MPI_Comm comm              = icComm->communicator();
   int rank                   = icComm->commRank();
   const int rcvProc          = 0;
#endif // PV_USE_MPI

   int nk;
   float const *baseBuffer;
   resetStats();

   nk = getTargetLayer()->getNumNeurons();

   int nbatch = getTargetLayer()->getLayerLoc()->nbatch;

   comptimer->start();
   switch (type) {
      case BufV:
         baseBuffer = retrieveVBuffer();
         for (int b = 0; b < nbatch; b++) {
            float const *buf = baseBuffer + b * getTargetLayer()->getNumNeurons();
            if (buf == NULL) {
#ifdef PV_USE_MPI
               if (rank != rcvProc) {
                  return Response::SUCCESS;
               }
#endif // PV_USE_MPI
               output(b) << getMessage() << "V buffer is NULL\n";
               return Response::SUCCESS;
            }
            for (int k = 0; k < nk; k++) {
               float a = buf[k];
               sum[b] += (double)a;
               sum2[b] += (double)(a * a);
               if (fabsf(a) > nnzThreshold) {
                  nnz[b]++;
               }
               if (a < fMin[b]) {
                  fMin[b] = a;
               }
               if (a > fMax[b]) {
                  fMax[b] = a;
               }
            }
         }
         break;
      case BufActivity:
         baseBuffer = retrieveActivityBuffer();
         for (int b = 0; b < nbatch; b++) {
            float const *buf = baseBuffer + b * getTargetLayer()->getNumExtended();
            for (int k = 0; k < nk; k++) {
               const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
               int kex               = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.dn,
                     loc->halo.up); // TODO: faster to use strides
               // and a double-loop than
               // compute
               // kIndexExtended for every neuron?
               float a = buf[kex];
               sum[b] += (double)a;
               sum2[b] += (double)(a * a);
               if (fabsf(a) > nnzThreshold) {
                  nnz[b]++;
               }
               if (a < fMin[b]) {
                  fMin[b] = a;
               }
               if (a > fMax[b]) {
                  fMax[b] = a;
               }
            }
         }
         break;
      default: pvAssert(0); break;
   }
   comptimer->stop();

#ifdef PV_USE_MPI
   mpitimer->start();
   int ierr;

   // In place reduction to prevent allocating a temp recv buffer
   ierr = MPI_Allreduce(MPI_IN_PLACE, sum, nbatch, MPI_DOUBLE, MPI_SUM, comm);
   ierr = MPI_Allreduce(MPI_IN_PLACE, sum2, nbatch, MPI_DOUBLE, MPI_SUM, comm);
   ierr = MPI_Allreduce(MPI_IN_PLACE, nnz, nbatch, MPI_INT, MPI_SUM, comm);
   ierr = MPI_Allreduce(MPI_IN_PLACE, fMin, nbatch, MPI_FLOAT, MPI_MIN, comm);
   ierr = MPI_Allreduce(MPI_IN_PLACE, fMax, nbatch, MPI_FLOAT, MPI_MAX, comm);
   ierr = MPI_Allreduce(MPI_IN_PLACE, &nk, 1, MPI_INT, MPI_SUM, comm);

   mpitimer->stop();
   if (rank != rcvProc) {
      return Response::SUCCESS;
   }

#endif // PV_USE_MPI
   float divisor = nk;

   iotimer->start();
   for (int b = 0; b < nbatch; b++) {
      avg[b]              = (float)sum[b] / divisor;
      sigma[b]            = sqrtf((float)sum2[b] / divisor - avg[b] * avg[b]);
      float avgval        = 0.0f;
      char const *avgnote = nullptr;
      if (type == BufActivity
          and getTargetLayer()->getComponentByType<BasePublisherComponent>()->getSparseLayer()) {
         avgval  = 1000.0f * avg[b]; // convert spikes per millisecond to hertz.
         avgnote = " Hz (/dt ms)";
      }
      else {
         avgval  = avg[b];
         avgnote = "";
      }
      output(b).printf(
            "%st==%6.1f b==%d N==%d Total==%f Min==%f Avg==%f%s "
            "Max==%f sigma==%f nnz==%d",
            getMessage(),
            (double)simTime,
            (int)b,
            (int)divisor,
            (double)sum[b],
            (double)fMin[b],
            (double)avgval,
            avgnote,
            (double)fMax[b],
            (double)sigma[b],
            (int)nnz[b]);
      output(b) << std::endl;
   }

   iotimer->stop();

   return Response::SUCCESS;
}

float const *StatsProbe::retrieveVBuffer() {
   auto *activityComponent = getTargetLayer()->getComponentByType<ActivityComponent>();
   auto *internalState     = activityComponent->getComponentByType<InternalStateBuffer>();
   FatalIf(
         internalState == nullptr,
         "%s target layer \"%s\" does not have a V buffer.\n",
         getDescription_c(),
         getTargetLayer()->getName());
   return internalState->getBufferData();
}

float const *StatsProbe::retrieveActivityBuffer() {
   auto *publisherComponent = getTargetLayer()->getComponentByType<BasePublisherComponent>();
   FatalIf(
         publisherComponent == nullptr,
         "%s target layer \"%s\" does not have an A buffer.\n",
         getDescription_c(),
         getTargetLayer()->getName());
   return publisherComponent->getLayerData();
}

int StatsProbe::checkpointTimers(PrintStream &timerstream) {
   iotimer->fprint_time(timerstream);
   mpitimer->fprint_time(timerstream);
   comptimer->fprint_time(timerstream);
   return PV_SUCCESS;
}
}
