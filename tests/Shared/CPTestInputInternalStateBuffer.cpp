/*
 * CPTestInputInternalStateBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "CPTestInputInternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

CPTestInputInternalStateBuffer::CPTestInputInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

CPTestInputInternalStateBuffer::~CPTestInputInternalStateBuffer() {}

void CPTestInputInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   GSynInternalStateBuffer::initialize(name, params, comm);
}

void CPTestInputInternalStateBuffer::setObjectType() {
   mObjectType = "CPTestInputInternalStateBuffer";
}

Response::Status CPTestInputInternalStateBuffer::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   auto status = GSynInternalStateBuffer::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Initially, each neuron's value is its global neuron index
   const PVLayerLoc *loc = getLayerLoc();
   for (int b = 0; b < loc->nbatch; b++) {
      float *VBatch = mBufferData.data() + b * getBufferSize();
      for (int k = 0; k < getBufferSize(); k++) {
         int kx = kxPos(k, loc->nx, loc->nx, loc->nf);
         int ky = kyPos(k, loc->nx, loc->ny, loc->nf);
         int kf = featureIndex(k, loc->nx, loc->ny, loc->nf);
         int kGlobal =
               kIndex(loc->kx0 + kx, loc->ky0 + ky, kf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         VBatch[k] = (float)kGlobal;
      }
   }

   return Response::SUCCESS;
}

void CPTestInputInternalStateBuffer::updateBufferCPU(double simTime, double deltaTime) {
   int const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   float *V                        = mBufferData.data();
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      V[k] += 1.0f;
   }
}

} // namespace PV
