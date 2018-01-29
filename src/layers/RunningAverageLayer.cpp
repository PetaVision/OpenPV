/*
 * RunningAverageLayer.cpp
 *
 *  Created on: Mar 3, 2015
 *      Author: wchavez
 */

#include "RunningAverageLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

namespace PV {
RunningAverageLayer::RunningAverageLayer() { initialize_base(); }

RunningAverageLayer::RunningAverageLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

RunningAverageLayer::~RunningAverageLayer() {}

int RunningAverageLayer::initialize_base() {
   originalLayer      = NULL;
   numImagesToAverage = 10;
   numUpdateTimes     = 0;
   return PV_SUCCESS;
}

int RunningAverageLayer::initialize(const char *name, HyPerCol *hc) {
   int status_init = CloneVLayer::initialize(name, hc);
   return status_init;
}

Response::Status RunningAverageLayer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return CloneVLayer::communicateInitInfo(message);
   // CloneVLayer sets originalLayer and errors out if originalLayerName is not valid
}

// RunningAverageLayer does not use the V buffer, so absolutely fine to clone off of an null V layer
void RunningAverageLayer::allocateV() {
   // Do nothing
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
   parent->parameters()->ioParamValue(
         ioFlag, name, "numImagesToAverage", &numImagesToAverage, numImagesToAverage);
}

int RunningAverageLayer::setActivity() {
   float *activity = clayer->activity->data;
   memset(activity, 0, sizeof(float) * clayer->numExtendedAllBatches);
   return 0;
}

Response::Status RunningAverageLayer::updateState(double timef, double dt) {
   numUpdateTimes++;
   // Check if an update is needed
   // Done in cloneVLayer
   int numNeurons                = originalLayer->getNumNeurons();
   float *A                      = clayer->activity->data;
   const float *originalA        = originalLayer->getCLayer()->activity->data;
   const PVLayerLoc *loc         = getLayerLoc();
   const PVLayerLoc *locOriginal = originalLayer->getLayerLoc();
   int nbatch                    = loc->nbatch;
   // Make sure all sizes match
   assert(locOriginal->nx == loc->nx);
   assert(locOriginal->ny == loc->ny);
   assert(locOriginal->nf == loc->nf);

   for (int b = 0; b < nbatch; b++) {
      const float *originalABatch = originalA + b * originalLayer->getNumExtended();
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
