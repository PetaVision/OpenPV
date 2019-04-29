/*
 * RetinaActivityBuffer.cpp
 *
 *  Created on: Jul 29, 2008
 */

#include "RetinaActivityBuffer.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "checkpointing/CheckpointEntryRandState.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

RetinaActivityBuffer::RetinaActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

RetinaActivityBuffer::~RetinaActivityBuffer() { delete mRandState; }

void RetinaActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

void RetinaActivityBuffer::setObjectType() { mObjectType = "RetinaActivityBuffer"; }

int RetinaActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_spikingFlag(ioFlag);
   ioParam_backgroundRate(ioFlag);
   ioParam_foregroundRate(ioFlag);
   ioParam_beginStim(ioFlag);
   ioParam_endStim(ioFlag);
   ioParam_burstFreq(ioFlag);
   ioParam_burstDuration(ioFlag);
   ioParam_refractoryPeriod(ioFlag);
   ioParam_absRefractoryPeriod(ioFlag);
   return status;
}

void RetinaActivityBuffer::ioParam_spikingFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "spikingFlag", &mSpikingFlag, true);
}

void RetinaActivityBuffer::ioParam_backgroundRate(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "backgroundRate", &mBackgroundRate, 0.0);
}

void RetinaActivityBuffer::ioParam_foregroundRate(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "foregroundRate", &mForegroundRate, 1.0);
}

void RetinaActivityBuffer::ioParam_beginStim(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "beginStim", &mBeginStim, 0.0);
}

void RetinaActivityBuffer::ioParam_endStim(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "endStim", &mEndStim, (double)FLT_MAX);
   if (ioFlag == PARAMS_IO_READ && mEndStim < 0)
      mEndStim = FLT_MAX;
}

void RetinaActivityBuffer::ioParam_burstFreq(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "burstFreq", &mBurstFreq, 1.0f);
}

void RetinaActivityBuffer::ioParam_burstDuration(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "burstDuration", &mBurstDuration, 1000.0f);
}

void RetinaActivityBuffer::ioParam_refractoryPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "spikingFlag"));
   if (mSpikingFlag) {
      parameters()->ioParamValue(
            ioFlag, name, "refractoryPeriod", &mRefractoryPeriod, mDefaultRefractoryPeriod);
   }
}

void RetinaActivityBuffer::ioParam_absRefractoryPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "spikingFlag"));
   if (mSpikingFlag) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "absRefractoryPeriod",
            &mAbsRefractoryPeriod,
            mDefaultAbsRefractoryPeriod);
   }
}

Response::Status RetinaActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mLayerInput = message->mObjectTable->findObject<LayerInputBuffer>(getName());
   FatalIf(
         mLayerInput == nullptr,
         "%s could not find an LayerInputBuffer component.\n",
         getDescription_c());
   mLayerInput->requireChannel(CHANNEL_EXC);
   mLayerInput->requireChannel(CHANNEL_INH);

   return Response::SUCCESS;
}

Response::Status RetinaActivityBuffer::allocateDataStructures() {
   checkDimensionsEqual(mLayerInput, this);
   auto status = ActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (mSpikingFlag) {
      mSinceLastSpike.resize(getBufferSizeAcrossBatch());
      // a random variable is needed for every neuron/clthread
      const PVLayerLoc *loc = getLayerLoc();
      mRandState            = new Random(loc, true /*extended*/);
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

Response::Status RetinaActivityBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ActivityBuffer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mSpikingFlag) {
      pvAssert(mRandState != nullptr);
      auto *checkpointer = message->mDataRegistry;
      registerRandState(checkpointer);
      registerSinceLastSpike(checkpointer);
   }
   return Response::SUCCESS;
}

void RetinaActivityBuffer::registerRandState(Checkpointer *checkpointer) {
   auto checkpointEntry = std::make_shared<CheckpointEntryRandState>(
         getName(),
         "rand_state",
         checkpointer->getMPIBlock(),
         mRandState->getRNG(0),
         getLayerLoc(),
         true /*extended buffer*/);
   bool registerSucceeded =
         checkpointer->registerCheckpointEntry(checkpointEntry, false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s buffer for checkpointing.\n",
         getDescription_c(),
         "rand_state");
}

void RetinaActivityBuffer::registerSinceLastSpike(Checkpointer *checkpointer) {
   auto checkpointEntry = std::make_shared<CheckpointEntryPvpBuffer<float>>(
         getName(),
         "SinceLastSpike",
         checkpointer->getMPIBlock(),
         mSinceLastSpike.data(),
         getLayerLoc(),
         true /*extended buffer*/);
   bool registerSucceeded =
         checkpointer->registerCheckpointEntry(checkpointEntry, false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s buffer for checkpointing.\n",
         getDescription_c(),
         "SinceLastSpike");
}

Response::Status
RetinaActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   setRetinaParams(message->mDeltaTime);
   if (mSpikingFlag) {
      int const numExtendedAcrossBatch = (int)mSinceLastSpike.size();
      for (int k = 0; k < numExtendedAcrossBatch; k++) {
         // allow neuron to fire at t=0
         mSinceLastSpike[k] = 10 * mRetinaParams.mAbsRefractoryPeriod;
      }
   }
   updateBufferCPU(0.0, message->mDeltaTime);
   return Response::SUCCESS;
}

void RetinaActivityBuffer::setRetinaParams(double deltaTime) {
   double dt_sec           = deltaTime * 0.001; // convert millisectonds to seconds
   mRetinaParams.mProbStim = (float)(mForegroundRate * dt_sec);
   if (mRetinaParams.mProbStim > 1.0f) {
      mRetinaParams.mProbStim = 1.0f;
   }
   mRetinaParams.mProbBase = (float)(mBackgroundRate * dt_sec);
   if (mRetinaParams.mProbBase > 1.0f) {
      mRetinaParams.mProbBase = 1.0f;
   }
   mRetinaParams.mBeginStim           = mBeginStim;
   mRetinaParams.mEndStim             = mEndStim;
   mRetinaParams.mBurstFreq           = mBurstFreq;
   mRetinaParams.mBurstDuration       = mBurstDuration;
   mRetinaParams.mRefractoryPeriod    = mRefractoryPeriod;
   mRetinaParams.mAbsRefractoryPeriod = mAbsRefractoryPeriod;
}

Response::Status RetinaActivityBuffer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   auto status = ActivityBuffer::readStateFromCheckpoint(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   readRandStateFromCheckpoint(checkpointer);
   readSinceLastSpikeFromCheckpoint(checkpointer);
   return Response::SUCCESS;
}

void RetinaActivityBuffer::readRandStateFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(
         std::string(name), std::string("rand_state.pvp"), false /*not constant*/);
}

void RetinaActivityBuffer::readSinceLastSpikeFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(
         std::string(name), std::string("SinceLastSpike.pvp"), false /*not constant*/);
}

void RetinaActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *loc = getLayerLoc();
   const int nx          = loc->nx;
   const int ny          = loc->ny;
   const int nf          = loc->nf;
   const int nbatch      = loc->nbatch;
   const int numNeurons  = nx * ny * nf;

   pvAssert(mLayerInput->getNumChannels() >= 2); // CHANNEL_EXC and CHANNEL_INH
   float const *GSynHead = mLayerInput->getBufferData();
   float *activity       = getReadWritePointer();

   if (mSpikingFlag) {
      spikingUpdateBuffer(
            nbatch,
            numNeurons,
            simTime,
            deltaTime,
            nx,
            ny,
            nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up,
            &mRetinaParams,
            mRandState->getRNG(0),
            GSynHead,
            activity,
            mSinceLastSpike.data());
   }
   else {
      nonspikingUpdateBuffer(
            nbatch,
            numNeurons,
            simTime,
            deltaTime,
            nx,
            ny,
            nf,
            loc->halo.lt,
            loc->halo.rt,
            loc->halo.dn,
            loc->halo.up,
            &mRetinaParams,
            GSynHead,
            activity);
   }
}

float RetinaActivityBuffer::calcBurstStatus(double simTime, RetinaParams *retinaParams) {
   float burstStatus;
   if (retinaParams->mBurstDuration <= 0 || retinaParams->mBurstFreq == 0) {
      burstStatus = cosf(2.0f * PI * (float)simTime * retinaParams->mBurstFreq / 1000.0f);
   }
   else {
      burstStatus = fmodf((float)simTime, 1000.0f / retinaParams->mBurstFreq);
      burstStatus = burstStatus < retinaParams->mBurstDuration;
   }
   burstStatus *=
         (int)((simTime >= retinaParams->mBeginStim) && (simTime < retinaParams->mEndStim));
   return burstStatus;
}

int RetinaActivityBuffer::spike(
      float simTime,
      float deltaTime,
      float timeSinceLast,
      float stimFactor,
      taus_uint4 *rnd_state,
      float burstStatus,
      RetinaParams *retinaParams) {
   float probSpike;

   // input parameters
   //
   float probBase = retinaParams->mProbBase;
   float probStim = retinaParams->mProbStim * stimFactor;

   // see if neuron is in a refractory period
   //
   if (timeSinceLast < retinaParams->mAbsRefractoryPeriod) {
      return 0;
   }
   else {
      float delta   = timeSinceLast - retinaParams->mAbsRefractoryPeriod;
      float refract = 1.0f - expf(-delta / retinaParams->mRefractoryPeriod);
      refract       = (refract < 0) ? 0 : refract;
      probBase *= refract;
      probStim *= refract;
   }

   probSpike = probBase;

   probSpike += probStim * burstStatus; // negative prob is OK
   // probSpike is spikes per millisecond; conversion to expected number of spikes in deltaTime
   // takes place
   // in setRetinaParams

   *rnd_state     = cl_random_get(*rnd_state);
   int spike_flag = (cl_random_prob(*rnd_state) < probSpike);
   return spike_flag;
}

void RetinaActivityBuffer::spikingUpdateBuffer(
      const int nbatch,
      const int numNeurons,
      const double simTime,
      const double deltaTime,

      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      RetinaParams *retinaParams,
      taus_uint4 *rnd,
      float const *GSynHead,
      float *activity,
      float *timeSinceLast) {

   float const *phiExc = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
   float const *phiInh = &GSynHead[CHANNEL_INH * nbatch * numNeurons];
   for (int b = 0; b < nbatch; b++) {
      taus_uint4 *rndBatch      = rnd + b * nx * ny * nf;
      float const *phiExcBatch  = phiExc + b * nx * ny * nf;
      float const *phiInhBatch  = phiInh + b * nx * ny * nf;
      float *timeSinceLastBatch = timeSinceLast + b * (nx + lt + rt) * (ny + up + dn) * nf;
      float *activityBatch      = activity + b * (nx + lt + rt) * (ny + up + dn) * nf;
      int k;
      float burstStatus = calcBurstStatus(simTime, retinaParams);
      for (k = 0; k < nx * ny * nf; k++) {
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         //
         // kernel (nonheader part) begins here
         //
         // load local variables from global memory
         //
         taus_uint4 l_rnd      = rndBatch[k];
         float l_phiExc        = phiExcBatch[k];
         float l_phiInh        = phiInhBatch[k];
         float l_timeSinceLast = timeSinceLastBatch[kex] + (float)deltaTime;
         float l_activ;
         l_activ = (float)spike(
               (float)simTime,
               (float)deltaTime,
               l_timeSinceLast,
               (l_phiExc - l_phiInh),
               &l_rnd,
               burstStatus,
               retinaParams);
         l_timeSinceLast = (l_activ > 0.0f) ? 0.0f : l_timeSinceLast;
         // store local variables back to global memory
         //
         rndBatch[k]        = l_rnd;
         timeSinceLast[kex] = l_timeSinceLast;
         activityBatch[kex] = l_activ;
      }
   }
}

void RetinaActivityBuffer::nonspikingUpdateBuffer(
      const int nbatch,
      const int numNeurons,
      const double simTime,
      const double deltaTime,

      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      RetinaParams *retinaParams,
      float const *GSynHead,
      float *activity) {
   int k;
   float burstStatus = calcBurstStatus(simTime, retinaParams);

   float const *phiExc = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
   float const *phiInh = &GSynHead[CHANNEL_INH * nbatch * numNeurons];

   for (int b = 0; b < nbatch; b++) {
      float const *phiExcBatch = phiExc + b * nx * ny * nf;
      float const *phiInhBatch = phiInh + b * nx * ny * nf;
      float *activityBatch     = activity + b * (nx + lt + rt) * (ny + up + dn) * nf;
      for (k = 0; k < nx * ny * nf; k++) {
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         //
         // kernel (nonheader part) begins here
         //

         // load local variables from global memory
         //
         float l_phiExc = phiExcBatch[k];
         float l_phiInh = phiInhBatch[k];
         float l_activ;
         // adding base prob should not change default behavior
         l_activ = burstStatus * retinaParams->mProbStim * (l_phiExc - l_phiInh)
                   + retinaParams->mProbBase;
         // store local variables back to global memory
         //
         activityBatch[kex] = l_activ;
      }
   }
}

} // namespace PV
