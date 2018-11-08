/*
 * RunningAverageLayer.cpp
 *
 *  Created on: Mar 3, 2015
 *      Author: wchavez
 */

// RunningAverageLayer was deprecated on Aug 15, 2018.

#include "RunningAverageLayer.hpp"
#include "include/default_params.h"
#include <stdio.h>

namespace PV {
RunningAverageLayer::RunningAverageLayer() { initialize_base(); }

RunningAverageLayer::RunningAverageLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

RunningAverageLayer::~RunningAverageLayer() {}

int RunningAverageLayer::initialize_base() {
   numImagesToAverage = 10;
   numUpdateTimes     = 0;
   return PV_SUCCESS;
}

void RunningAverageLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   WarnLog() << "RunningAverageLayer has been deprecated.\n";
   int status_init = CloneVLayer::initialize(name, params, comm);
   return status_init;
}

Response::Status RunningAverageLayer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return CloneVLayer::communicateInitInfo(message);
   // CloneVLayer sets mOriginalLayer and errors out if originalLayerName is not valid
}

int RunningAverageLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_numImagesToAverage(ioFlag);

   if (numImagesToAverage <= 0) {
      Fatal().printf(
            "RunningAverageLayer: numImagesToAverage must be an integer greater than 0.\n");
   }
   return PV_SUCCESS;
}

void RunningAverageLayer::ioParam_numImagesToAverage(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "numImagesToAverage", &numImagesToAverage, numImagesToAverage);
}

int RunningAverageLayer::setActivity() {
   float *activity = mActivity->getActivity();
   memset(activity, 0, sizeof(float) * getNumExtendedAllBatches());
   return 0;
}

Response::Status RunningAverageLayer::updateState(double timef, double dt) {
   numUpdateTimes++;
   // Check if an update is needed
   // Done in cloneVLayer
   int numNeurons                = mOriginalLayer->getNumNeurons();
   float *A                      = mActivity->getActivity();
   const float *originalA        = mOriginalLayer->getActivity();
   const PVLayerLoc *loc         = getLayerLoc();
   const PVLayerLoc *locOriginal = mOriginalLayer->getLayerLoc();
   int nbatch                    = loc->nbatch;
   // Make sure all sizes match
   assert(locOriginal->nx == loc->nx);
   assert(locOriginal->ny == loc->ny);
   assert(locOriginal->nf == loc->nf);

   for (int b = 0; b < nbatch; b++) {
      const float *originalABatch = originalA + b * mOriginalLayer->getNumExtended();
      float *ABatch               = A + b * getNumExtended();
      if (numUpdateTimes < numImagesToAverage * dt) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < numNeurons; k++) {
            int kExt = kIndexExtended(
                  k,
                  loc->nx,
                  loc->ny,
                  loc->nf,
                  loc->halo.lt,
                  loc->halo.rt,
                  loc->halo.dn,
                  loc->halo.up);
            int kExtOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  loc->halo.rt,
                  loc->halo.dn,
                  loc->halo.up);
            ABatch[kExt] =
                  ((numUpdateTimes / (float)dt - 1) * ABatch[kExt] + originalABatch[kExtOriginal])
                  * (float)dt / numUpdateTimes;
         }
      }
      else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < numNeurons; k++) {
            int kExt = kIndexExtended(
                  k,
                  loc->nx,
                  loc->ny,
                  loc->nf,
                  loc->halo.lt,
                  loc->halo.rt,
                  loc->halo.dn,
                  loc->halo.up);
            int kExtOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  loc->halo.rt,
                  loc->halo.dn,
                  loc->halo.up);
            ABatch[kExt] = ((numImagesToAverage - 1) * ABatch[kExt] + originalABatch[kExtOriginal])
                           / numImagesToAverage;
         }
      }
   }
   return Response::SUCCESS;
}

} // end namespace PV
