/*
 * RescaleLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
#include "RescaleLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

namespace PV {
RescaleLayer::RescaleLayer() {
   initialize_base();
}

RescaleLayer::RescaleLayer(const char * name, HyPerCol * hc, HyPerLayer * originalLayer) {
   initialize_base();
   initialize(name, hc, originalLayer);
}

RescaleLayer::~RescaleLayer()
{
    clayer->V = NULL;
}

int RescaleLayer::initialize_base() {
   sourceLayer = NULL;
   setMax = 1;
   setMin = -1;
   return PV_SUCCESS;
}

int RescaleLayer::initialize(const char * name, HyPerCol * hc, HyPerLayer * clone) {
   //int num_channels = sourceLayer->getNumChannels();
   int status_init = HyPerLayer::initialize(name, hc, 0);

   sourceLayer = clone;
   free(clayer->V);
   clayer->V = sourceLayer->getV();

   // don't need conductance channels
   freeChannels();

   return status_init;
}

int RescaleLayer::setParams(PVParams * params){
   HyPerLayer::setParams(params);
   readSetMax(params);
   readSetMin(params);
   return PV_SUCCESS;
}

void RescaleLayer::readSetMax(PVParams * params){
   setMax = params->value(name, "setMax", setMax);
}

void RescaleLayer::readSetMin(PVParams * params){
   setMin = params->value(name, "setMin", setMin);
}

int RescaleLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

int RescaleLayer::updateState(double timef, double dt) {
   int status;
   float maxV = -1000000000;
   float minV = 1000000000;
   int numNeurons = sourceLayer->getNumNeurons();
   pvdata_t * V = clayer->V; 
   pvdata_t * A = getActivity();
   const PVLayerLoc * loc = getLayerLoc();
   
   //Find max and min of V
   for (int k = 0; k < numNeurons; k++){
      if (V[k] > maxV){
         maxV = V[k];
      }
      if (V[k] < minV){
         minV = V[k];
      }
   }

#ifdef PV_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &maxV, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &minV, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI

   float rangeV = maxV - minV;
   for (int k = 0; k < numNeurons; k++){
      int kext = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
      A[kext] = ((V[k] - minV)/rangeV) * (setMax - setMin) + setMin;
   }

   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

