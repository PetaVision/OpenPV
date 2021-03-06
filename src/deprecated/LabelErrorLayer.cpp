/*
 * LabelErrorLayer.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: garkenyon
 */

// LabelErrorLayer was deprecated on Aug 15, 2018.

#include "LabelErrorLayer.hpp"
#include "DeprecatedUpdateStateFunctions.h"

void LabelErrorLayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      float *GSynHead,
      float *activity,
      float errScale,
      int isBinary);

namespace PV {

LabelErrorLayer::LabelErrorLayer() { initialize_base(); }

LabelErrorLayer::LabelErrorLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

LabelErrorLayer::~LabelErrorLayer() {}

int LabelErrorLayer::initialize_base() {
   errScale = 1;
   isBinary = 1;
   return PV_SUCCESS;
}

void LabelErrorLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   WarnLog() << "LabelErrorLayer has been deprecated.\n";
   int status = ANNLayer::initialize(name, params, comm);
   mLayerInput->requireChannel(1);
   assert(mLayerInput->getNumChannels() == 2);
   return status;
}

int LabelErrorLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_errScale(ioFlag);
   ioParam_isBinary(ioFlag);
   return status;
}

void LabelErrorLayer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "errScale", &errScale, errScale);
}

void LabelErrorLayer::ioParam_isBinary(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "isBinary", &isBinary, isBinary);
}

Response::Status LabelErrorLayer::updateState(double time, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = mActivity->getActivity();
   float *V              = getV();
   int num_channels      = mLayerInput->getNumChannels();
   float *gSynHead       = mLayerInput->getLayerInput();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   LabelErrorLayer_update_state(
         nbatch,
         num_neurons,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         V,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         gSynHead,
         A,
         errScale,
         isBinary);

   return Response::SUCCESS;
}

} /* namespace PV */

// Kernel

void LabelErrorLayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,

      float *V,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      float *GSynHead,
      float *activity,
      const float errScale,
      const int isBinary) {
   updateV_LabelErrorLayer(
         nbatch,
         numNeurons,
         V,
         GSynHead,
         activity,
         numVertices,
         verticesV,
         verticesA,
         slopes,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up,
         errScale,
         isBinary);
}
