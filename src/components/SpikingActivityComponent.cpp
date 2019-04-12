/*
 * SpikingActivityComponent.cpp
 *
 *  Created on: Sep 12, 2018
 *      Author: twatkins
 */

#include "SpikingActivityComponent.hpp"
#include <cmath>

namespace PV {

SpikingActivityComponent::SpikingActivityComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

SpikingActivityComponent::~SpikingActivityComponent() {}

void SpikingActivityComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   SpikingActivityComponentBase::initialize(name, params, comm);
}

void SpikingActivityComponent::setObjectType() { mObjectType = "SpikingActivityComponent"; }

int SpikingActivityComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = SpikingActivityComponentBase::ioParamsFillGroup(ioFlag);
   ioParam_integrationTime(ioFlag);
   ioParam_VThresh(ioFlag);
   return status;
}

void SpikingActivityComponent::ioParam_integrationTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "integrationTime",
         &mIntegrationTime,
         mIntegrationTime,
         true /*warnIfAbsent*/);
}

void SpikingActivityComponent::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, name, "integrationTime", &mIntegrationTime);
}

Response::Status SpikingActivityComponent::updateActivity(double simTime, double deltaTime) {
   if (mAccumulatedGSyn) {
      mAccumulatedGSyn->updateBuffer(simTime, deltaTime);
   }
   float const *gSyn = mAccumulatedGSyn->getBufferData();
   float *V          = mInternalState->getReadWritePointer();
   float *A          = mActivity->getReadWritePointer();
   pvAssert(gSyn != nullptr and V != nullptr and A != nullptr);

   float const decayfactor = (float)std::exp(-deltaTime / (double)mIntegrationTime);
   PVLayerLoc const *loc   = getLayerLoc();
   int const nb            = loc->nbatch;
   int const nx            = loc->nx;
   int const ny            = loc->ny;
   int const nf            = loc->nf;
   int const lt            = loc->halo.lt;
   int const rt            = loc->halo.rt;
   int const dn            = loc->halo.dn;
   int const up            = loc->halo.up;
   int const numNeurons    = mAccumulatedGSyn->getBufferSizeAcrossBatch();
   pvAssert(mInternalState->getBufferSizeAcrossBatch() == numNeurons);
   pvAssert(mActivity->getBufferSizeAcrossBatch() >= numNeurons);
   for (int k = 0; k < numNeurons; k++) {
      V[k]     = V[k] * decayfactor + gSyn[k] * (1 - decayfactor);
      int kExt = kIndexExtendedBatch(k, nb, nx, ny, nf, lt, rt, dn, up);
      if (V[k] > mVThresh) {
         V[k]    = 0.0f;
         A[kExt] = 1.0f;
      }
      else {
         A[kExt] = 0.0f;
      }
   }

   return Response::SUCCESS;
}

} // namespace PV
