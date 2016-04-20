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
RunningAverageLayer::RunningAverageLayer() {
   initialize_base();
}

RunningAverageLayer::RunningAverageLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

RunningAverageLayer::~RunningAverageLayer()
{
   // Handled by CloneVLayer destructor
   // free(originalLayerName);
   // clayer->V = NULL;
}

int RunningAverageLayer::initialize_base() {
   originalLayer = NULL;
   numImagesToAverage = 10;
   numUpdateTimes = 0;
   return PV_SUCCESS;
}

int RunningAverageLayer::initialize(const char * name, HyPerCol * hc) {
   //int num_channels = sourceLayer->getNumChannels();
   int status_init = CloneVLayer::initialize(name, hc);
   return status_init;
}

int RunningAverageLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      fprintf(stderr, "Group \"%s\": Original layer \"%s\" must be a HyPer layer\n", name, originalLayerName);
   }
   return status;
}

//RunningAverageLayer does not use the V buffer, so absolutely fine to clone off of an null V layer
int RunningAverageLayer::allocateV() {
   //Do nothing
   return PV_SUCCESS;
}


int RunningAverageLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag){
  //readOriginalLayerName(params);  // done in CloneVLayer

   CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_numImagesToAverage(ioFlag);

   if(numImagesToAverage <= 0){
      fprintf(stderr, "RunningAverageLayer: numImagesToAverage must be an integer greater than 0.\n");
      exit(PV_FAILURE);
   }
   return PV_SUCCESS;

}

void RunningAverageLayer::ioParam_numImagesToAverage(enum ParamsIOFlag ioFlag){
   parent->ioParamValue(ioFlag, name, "numImagesToAverage", &numImagesToAverage, numImagesToAverage);
}

int RunningAverageLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtendedAllBatches);
   return 0;

}

int RunningAverageLayer::updateState(double timef, double dt) {
   numUpdateTimes++;
   int status = PV_SUCCESS;
   double deltaT = parent->getDeltaTime();
   //Check if an update is needed
   //Done in cloneVLayer
    int numNeurons = originalLayer->getNumNeurons();
    pvdata_t * A = clayer->activity->data;
    const pvdata_t * originalA = originalLayer->getCLayer()->activity->data;
    const PVLayerLoc * loc = getLayerLoc();
    const PVLayerLoc * locOriginal = originalLayer->getLayerLoc();
    int nbatch = loc->nbatch;
    //Make sure all sizes match
    //assert(locOriginal->nb == loc->nb);
    assert(locOriginal->nx == loc->nx);
    assert(locOriginal->ny == loc->ny);
    assert(locOriginal->nf == loc->nf);

    for(int b = 0; b < nbatch; b++){
       const pvdata_t * originalABatch = originalA + b * originalLayer->getNumExtended();
       pvdata_t * ABatch = A + b * getNumExtended();
       if (numUpdateTimes < numImagesToAverage*deltaT){
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
             for(int k=0; k<numNeurons; k++) {
                int kExt = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
                int kExtOriginal = kIndexExtended(k, locOriginal->nx, locOriginal->ny, locOriginal->nf,
                      locOriginal->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
                ABatch[kExt] = ((numUpdateTimes/deltaT-1) * ABatch[kExt] + originalABatch[kExtOriginal]) * deltaT / numUpdateTimes;
             }
       }
       else{
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
          for(int k=0; k<numNeurons; k++) {
             int kExt = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
             int kExtOriginal = kIndexExtended(k, locOriginal->nx, locOriginal->ny, locOriginal->nf,
                   locOriginal->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
             ABatch[kExt] = ((numImagesToAverage-1) * ABatch[kExt] + originalABatch[kExtOriginal]) / numImagesToAverage;
          }
       }
    }

    //Update lastUpdateTime
    lastUpdateTime = parent->simulationTime();

   return status;
}

BaseObject * createRunningAverageLayer(char const * name, HyPerCol * hc) {
   return hc ? new RunningAverageLayer(name, hc) : NULL;
}

} // end namespace PV

