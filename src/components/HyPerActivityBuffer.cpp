/*
 * HyPerActivityBuffer.cpp
 *
 *  Created on: Oct 12, 2018 from code from the original HyPerLayer
 *      Author: Pete Schultz
 */

#include "HyPerActivityBuffer.hpp"

namespace PV {

HyPerActivityBuffer::HyPerActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerActivityBuffer::~HyPerActivityBuffer() {}

void HyPerActivityBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   VInputActivityBuffer::initialize(name, params, comm);
}

void HyPerActivityBuffer::setObjectType() { mObjectType = "HyPerActivityBuffer"; }

Response::Status
HyPerActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (!mInternalState->getInitialValuesSetFlag()) {
      return Response::POSTPONE;
   }
   updateBufferCPU(0.0 /*simTime*/, message->mDeltaTime);
   return Response::SUCCESS;
}

void HyPerActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *A           = mBufferData.data();
   float const *V     = mInternalState->getBufferData();
   int const nbatch   = getLayerLoc()->nbatch;
   int const nx       = getLayerLoc()->nx;
   int const ny       = getLayerLoc()->ny;
   int const nf       = getLayerLoc()->nf;
   PVHalo const *halo = &getLayerLoc()->halo;

   int const numNeuronsAcrossBatch = mInternalState->getBufferSizeAcrossBatch();
   pvAssert(V != nullptr);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      int kExt = kIndexExtendedBatch(k, nbatch, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up);
      A[kExt]  = V[k];
   }
}

} // namespace PV
