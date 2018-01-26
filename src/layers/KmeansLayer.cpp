/*
 * KmeansLayer.cpp
 *
 *  Created on: Dec. 1, 2014
 *      Author: Xinhua Zhang
 */

#include "KmeansLayer.hpp"
#include "../layers/updateStateFunctions.h"

namespace PV {

KmeansLayer::KmeansLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

KmeansLayer::~KmeansLayer() {}

KmeansLayer::KmeansLayer() { initialize_base(); }

int KmeansLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   assert(status == PV_SUCCESS);
   return status;
}

int KmeansLayer::initialize_base() {
   numChannels  = 1;
   trainingFlag = false;
   return PV_SUCCESS;
}

Response::Status KmeansLayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = clayer->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();

   float *gSynHead = GSyn == NULL ? NULL : GSyn[0];
   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int num_neurons = nx * ny * nf;
   int nbatch      = loc->nbatch;

   setActivity_KmeansLayer(
         nbatch,
         num_neurons,
         num_channels,
         gSynHead,
         A,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         trainingFlag);

   return Response::SUCCESS;
}

int KmeansLayer::setActivity() {
   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   PVHalo const *halo    = &loc->halo;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   int status;

   status = setActivity_HyPerLayer(
         nbatch,
         num_neurons,
         getCLayer()->activity->data,
         getV(),
         nx,
         ny,
         nf,
         halo->lt,
         halo->rt,
         halo->dn,
         halo->up);

   return status;
}

void KmeansLayer::ioParam_TrainingFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "training", &trainingFlag, trainingFlag);
}

int KmeansLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_TrainingFlag(ioFlag);

   return status;
}

BaseObject *createKmeansLayer(char const *name, HyPerCol *hc) {
   return hc ? new KmeansLayer(name, hc) : NULL;
}
}
