/*
 * HyPerInternalStateBuffer.cpp
 *
 *  Created on: Oct 12, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#include "HyPerInternalStateBuffer.hpp"

namespace PV {

HyPerInternalStateBuffer::HyPerInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerInternalStateBuffer::~HyPerInternalStateBuffer() {}

void HyPerInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
}

void HyPerInternalStateBuffer::setObjectType() { mObjectType = "HyPerInternalStateBuffer"; }

Response::Status HyPerInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = InternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mAccumulatedGSyn = message->mObjectTable->findObject<GSynAccumulator>(getName());
   FatalIf(
         mAccumulatedGSyn == nullptr,
         "%s could not find a GSynAccumulator component.\n",
         getDescription_c());

   return Response::SUCCESS;
}

void HyPerInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *V = getReadWritePointer();
   if (V == nullptr) {
      WarnLog().printf(
            "%s is not updateable. updateBuffer called with t=%f, dt=%f.\n",
            getDescription_c(),
            simTime,
            deltaTime);
      return;
   }
   int const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   pvAssert(numNeuronsAcrossBatch == mAccumulatedGSyn->getBufferSizeAcrossBatch());

   float const *gSyn = mAccumulatedGSyn->getBufferData();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] = gSyn[k];
   }
}

} // namespace PV
