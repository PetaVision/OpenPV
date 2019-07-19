/*
 * WeightedOccludingInternalStateBuffer.cpp
 *
 *  Created on: Jul 19, 2019
 *      Author: Jacob Springer
 */

#include "WeightedOccludingInternalStateBuffer.hpp"

namespace PV {

WeightedOccludingInternalStateBuffer::WeightedOccludingInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

WeightedOccludingInternalStateBuffer::~WeightedOccludingInternalStateBuffer() {}

void WeightedOccludingInternalStateBuffer::ioParam_occludingLayerName(enum ParamsIOFlag ioFlag) {
   this->parameters()->ioParamString(
           ioFlag, this->getName(), "occludingLayerName", &mOccludingLayerName, NULL, true);
}

void WeightedOccludingInternalStateBuffer::ioParam_occlusionDepth(enum ParamsIOFlag ioFlag) {
   // TODO: ensure this parameter is in in the valid range
   this->parameters()->ioParamValue(
           ioFlag, this->getName(), "occlusionDepth", &mOcclusionDepth, mOcclusionDepth, true);
}

int WeightedOccludingInternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = InternalStateBuffer::ioParamsFillGroup(ioFlag);
   ioParam_occludingLayerName(ioFlag);
   return status;
}

void WeightedOccludingInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
}

void WeightedOccludingInternalStateBuffer::setObjectType() { mObjectType = "WeightedOccludingInternalStateBuffer"; }

Response::Status WeightedOccludingInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = InternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mAccumulatedGSyn = message->mObjectTable->findObject<GSynAccumulator>(getName());
   mOccludingGSyn = message->mObjectTable->findObject<OccludingGSynAccumulator>(mOccludingLayerName);
   FatalIf(
         mAccumulatedGSyn == nullptr,
         "%s could not find an GSynAccumulator component.\n",
         getDescription_c());
   FatalIf(
         mOccludingGSyn == nullptr,
         "%s could not find an OccludingGSynAccumulator component.\n",
         getDescription_c());

   return Response::SUCCESS;
}

void WeightedOccludingInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
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

   PVLayerLoc const *loc           = getLayerLoc();
   int const numPixelsAcrossBatch = loc->nbatch * loc->nx * loc->ny;
   float const *gSyn = mAccumulatedGSyn->getBufferData();
   float const *contrib = mOccludingGSyn->retrieveContribData();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] = gSyn[k] * contrib[mOcclusionDepth * numPixelsAcrossBatch + k / loc->nf];
   }
}

} // namespace PV
