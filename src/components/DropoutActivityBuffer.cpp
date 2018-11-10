/*
 * DropoutActivityBuffer.cpp
 *
 *  Created on: Nov 7, 2016
 *      Author: Austin Thresher
 */

#include "DropoutActivityBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

DropoutActivityBuffer::DropoutActivityBuffer(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

DropoutActivityBuffer::~DropoutActivityBuffer() {}

int DropoutActivityBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = ANNActivityBuffer::initialize(name, hc);
   return status;
}

void DropoutActivityBuffer::setObjectType() { mObjectType = "DropoutActivityBuffer"; }

int DropoutActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_probability(ioFlag);
   return ANNActivityBuffer::ioParamsFillGroup(ioFlag);
}

void DropoutActivityBuffer::ioParam_probability(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "probability", &mProbability, mProbability, true);
   if (mProbability < 0) {
      WarnLog() << getName() << ": probability was set to < 0%. Changing to 0%.\n";
      mProbability = 99;
   }
   if (mProbability > 99) {
      WarnLog() << getName() << ": probability was set to >= 100%. Changing to 99%.\n";
      mProbability = 99;
   }
}

void DropoutActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   ANNActivityBuffer::updateBufferCPU(simTime, deltaTime);
   float *A  = mBufferData.data();
   int total = getBufferSizeAcrossBatch();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int i = 0; i < total; ++i) {
      if (rand() % 100 < mProbability) {
         A[i] = 0.0f;
      }
   }
}

} // namespace PV
