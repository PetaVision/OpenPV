#include "ResetStateOnTriggerTestProbe.hpp"
#include <layers/HyPerLayer.hpp>

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe(char const *name, PV::HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe() { initialize_base(); }

int ResetStateOnTriggerTestProbe::initialize_base() {
   probeStatus      = 0;
   firstFailureTime = 0;
   return PV_SUCCESS;
}

int ResetStateOnTriggerTestProbe::initialize(char const *name, PV::HyPerCol *hc) {
   int status = PV_SUCCESS;
   status     = PV::LayerProbe::initialize(name, hc);
   return status;
}

void ResetStateOnTriggerTestProbe::calcValues(double timevalue) {
   int nBatch = getNumValues();
   if (timevalue > 0.0) {
      int N                 = targetLayer->getNumNeurons();
      int NGlobal           = targetLayer->getNumGlobalNeurons();
      PVLayerLoc const *loc = targetLayer->getLayerLoc();
      PVHalo const *halo    = &loc->halo;
      int inttime           = (int)nearbyintf(timevalue / parent->getDeltaTime());
      for (int b = 0; b < nBatch; b++) {
         int numDiscreps       = 0;
         float const *activity = targetLayer->getLayerData() + b * targetLayer->getNumExtended();
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
            parent->getCommunicator()->communicator());
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

PV::Response::Status ResetStateOnTriggerTestProbe::outputState(double timevalue) {
   getValues(timevalue); // calls calcValues
   if (mOutputStreams.empty()) {
      return PV::Response::SUCCESS;
   }
   if (probeStatus != 0) {
      int nBatch = getNumValues();
      pvAssert(nBatch == mOutputStreams.size());
      int batchOffset = nBatch * (getMPIBlock()->getStartBatch() + getMPIBlock()->getBatchIndex());
      int globalBatchSize = nBatch * getMPIBlock()->getGlobalBatchDimension();
      for (int localBatchIndex = 0; localBatchIndex < nBatch; localBatchIndex++) {
         int nnz = (int)nearbyint(getValuesBuffer()[localBatchIndex]);
         if (globalBatchSize == 1) {
            pvAssert(localBatchIndex == 0);
            output(localBatchIndex)
                  .printf(
                        "%s t=%f, %d neuron%s the wrong value.\n",
                        getMessage(),
                        timevalue,
                        nnz,
                        nnz == 1 ? " has" : "s have");
         }
         else {
            int globalBatchIndex = localBatchIndex + batchOffset;
            output(localBatchIndex)
                  .printf(
                        "%s t=%f, batch element %d, %d neuron%s the wrong value.\n",
                        getMessage(),
                        timevalue,
                        localBatchIndex,
                        nnz,
                        nnz == 1 ? " has" : "s have");
         }
      }
   }
   return PV::Response::SUCCESS;
}

ResetStateOnTriggerTestProbe::~ResetStateOnTriggerTestProbe() {}

PV::BaseObject *createResetStateOnTriggerTestProbe(char const *name, PV::HyPerCol *hc) {
   return hc ? new ResetStateOnTriggerTestProbe(name, hc) : NULL;
}
