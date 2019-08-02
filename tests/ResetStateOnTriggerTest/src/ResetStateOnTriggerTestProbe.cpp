#include "ResetStateOnTriggerTestProbe.hpp"
#include <components/BasePublisherComponent.hpp>
#include <layers/HyPerLayer.hpp>

using namespace PV;

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe() { initialize_base(); }

int ResetStateOnTriggerTestProbe::initialize_base() {
   probeStatus      = 0;
   firstFailureTime = 0;
   return PV_SUCCESS;
}

void ResetStateOnTriggerTestProbe::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   LayerProbe::initialize(name, params, comm);
}

Response::Status ResetStateOnTriggerTestProbe::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   mDeltaTime = message->mDeltaTime;
   return Response::SUCCESS;
}

void ResetStateOnTriggerTestProbe::calcValues(double timevalue) {
   int nBatch = getNumValues();
   if (timevalue > 0.0) {
      auto *targetPublisher = targetLayer->getComponentByType<BasePublisherComponent>();
      PVLayerLoc const *loc = targetLayer->getLayerLoc();
      int N                 = loc->nx * loc->ny * loc->nf;
      int NGlobal           = loc->nxGlobal * loc->nyGlobal * loc->nf;
      int numExtended       = targetLayer->getNumExtended();
      PVHalo const *halo    = &loc->halo;
      int inttime           = (int)nearbyintf(timevalue / mDeltaTime);
      for (int b = 0; b < nBatch; b++) {
         int numDiscreps       = 0;
         float const *activity = targetPublisher->getLayerData() + b * numExtended;
         for (int k = 0; k < N; k++) {
            int kex = kIndexExtended(
                  k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
            float a          = activity[kex];
            int kGlobal      = globalIndexFromLocal(k, *loc);
            int correctValue = 4 * kGlobal * ((inttime + 4) % 5 + 1)
                               + (kGlobal == ((((inttime - 1) / 5) * 5) + 1) % NGlobal);
            if (a != (float)correctValue) {
               numDiscreps++;
            }
         }
         getValuesBuffer()[b] = (double)numDiscreps;
      }
      MPI_Allreduce(
            MPI_IN_PLACE,
            getValuesBuffer(),
            nBatch,
            MPI_DOUBLE,
            MPI_SUM,
            mCommunicator->communicator());
      if (probeStatus == 0) {
         for (int k = 0; k < nBatch; k++) {
            if (getValuesBuffer()[k]) {
               probeStatus      = 1;
               firstFailureTime = timevalue;
            }
         }
      }
   }
   else {
      for (int b = 0; b < nBatch; b++) {
         getValuesBuffer()[b] = 0.0;
      }
   }
}

Response::Status ResetStateOnTriggerTestProbe::outputState(double simTime, double deltaTime) {
   getValues(simTime); // calls calcValues
   if (mOutputStreams.empty()) {
      return Response::SUCCESS;
   }
   if (probeStatus != 0) {
      int nBatch = getNumValues();
      pvAssert((std::size_t)nBatch == mOutputStreams.size());
      int globalBatchSize = nBatch * getMPIBlock()->getGlobalBatchDimension();
      for (int localBatchIndex = 0; localBatchIndex < nBatch; localBatchIndex++) {
         int nnz = (int)nearbyint(getValuesBuffer()[localBatchIndex]);
         if (globalBatchSize == 1) {
            pvAssert(localBatchIndex == 0);
            output(localBatchIndex)
                  .printf(
                        "%s t=%f, %d neuron%s the wrong value.\n",
                        getMessage(),
                        simTime,
                        nnz,
                        nnz == 1 ? " has" : "s have");
         }
         else {
            output(localBatchIndex)
                  .printf(
                        "%s t=%f, batch element %d, %d neuron%s the wrong value.\n",
                        getMessage(),
                        simTime,
                        localBatchIndex,
                        nnz,
                        nnz == 1 ? " has" : "s have");
         }
      }
   }
   return Response::SUCCESS;
}

ResetStateOnTriggerTestProbe::~ResetStateOnTriggerTestProbe() {}

BaseObject *
createResetStateOnTriggerTestProbe(char const *name, PVParams *params, Communicator const *comm) {
   return new ResetStateOnTriggerTestProbe(name, params, comm);
}
