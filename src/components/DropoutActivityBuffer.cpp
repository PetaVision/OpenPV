/*
 * DropoutActivityBuffer.cpp
 *
 *  Created on: Nov 7, 2016
 *      Author: Austin Thresher
 */

#include "DropoutActivityBuffer.hpp"

namespace PV {

DropoutActivityBuffer::DropoutActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DropoutActivityBuffer::~DropoutActivityBuffer() { delete mRandState; }

void DropoutActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ANNActivityBuffer::initialize(name, params, comm);
}

void DropoutActivityBuffer::setObjectType() { mObjectType = "DropoutActivityBuffer"; }

int DropoutActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_probability(ioFlag);
   return ANNActivityBuffer::ioParamsFillGroup(ioFlag);
}

void DropoutActivityBuffer::ioParam_probability(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "probability", &mProbability, mProbability, true);
   if (mProbability < 0) {
      WarnLog() << getName() << ": probability was set to < 0%. Changing to 0%.\n";
      mProbability = 99;
   }
   if (mProbability > 99) {
      WarnLog() << getName() << ": probability was set to >= 100%. Changing to 99%.\n";
      mProbability = 99;
   }
}

Response::Status DropoutActivityBuffer::allocateDataStructures() {
   auto status = ActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   mRandState = new Random(getLayerLoc(), true /*extended*/);
   return Response::SUCCESS;
}

void DropoutActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   ANNActivityBuffer::updateBufferCPU(simTime, deltaTime);
   float *A  = mBufferData.data();
   int total = getBufferSizeAcrossBatch();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int i = 0; i < total; ++i) {
      taus_uint4 *rng = mRandState->getRNG(i);
      *rng            = cl_random_get(*rng);
      if (rng->s0 % 100 < mProbability) {
         A[i] = 0.0f;
      }
   }
}

} // namespace PV
