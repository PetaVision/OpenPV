/*
 * KmeansLayer.cpp
 *
 *  Created on: Dec. 1, 2014
 *      Author: Xinhua Zhang
 */

// KmeansLayer was deprecated on Aug 15, 2018.

#include "KmeansLayer.hpp"
#include "DeprecatedUpdateStateFunctions.h"

namespace PV {

KmeansLayer::KmeansLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

KmeansLayer::~KmeansLayer() {}

KmeansLayer::KmeansLayer() { initialize_base(); }

void KmeansLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   WarnLog() << "KmeansLayer has been deprecated.\n";
   int status = HyPerLayer::initialize(name, params, comm);
   assert(status == PV_SUCCESS);
   return status;
}

int KmeansLayer::initialize_base() {
   trainingFlag = false;
   return PV_SUCCESS;
}

Response::Status KmeansLayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = mActivity->getActivity();
   float *V              = getV();
   int num_channels      = getNumChannels();

   float *gSynHead = mLayerInput->getLayerInput();
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
         getActivity(),
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
   parameters()->ioParamValue(ioFlag, name, "training", &trainingFlag, trainingFlag);
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
