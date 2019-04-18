#include "DropoutLayer.hpp"

namespace PV {

DropoutLayer::DropoutLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

DropoutLayer::~DropoutLayer() {}

int DropoutLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_probability(ioFlag);
   return ANNLayer::ioParamsFillGroup(ioFlag);
}

void DropoutLayer::ioParam_probability(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "probability", &mProbability, mProbability, true);
   if (mProbability > 99) {
      WarnLog() << getName() << ": probability was set to >= 100%. Changing to 99%.\n";
      mProbability = 99;
   }
}

Response::Status DropoutLayer::updateState(double timestamp, double dt) {
   ANNLayer::updateState(timestamp, dt);
   float *A  = getCLayer()->activity->data;
   int total = getNumExtendedAllBatches();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int i = 0; i < total; ++i) {
      if (rand() % 100 < mProbability) {
         A[i] = 0.0f;
      }
   }

   return Response::SUCCESS;
}
}
