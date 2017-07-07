/*
 * GapLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "GapLayer.hpp"
#include "HyPerLayer.hpp"

// GapLayer can be used to implement gap junctions
namespace PV {
GapLayer::GapLayer() { initialize_base(); }

GapLayer::GapLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

GapLayer::~GapLayer() {}

int GapLayer::initialize_base() {
   ampSpikelet = 50;
   return PV_SUCCESS;
}

int GapLayer::initialize(const char *name, HyPerCol *hc) {
   int status_init = CloneVLayer::initialize(name, hc);
   assert(originalLayerName != NULL);

   return status_init;
}

int GapLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_ampSpikelet(ioFlag);
   return status;
}

void GapLayer::ioParam_ampSpikelet(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "ampSpikelet", &ampSpikelet, ampSpikelet);
}

int GapLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = CloneVLayer::communicateInitInfo(message);

   // Handled by CloneVLayer

   return status;
}

int GapLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();
   return status;
}

int GapLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(
         timef,
         dt,
         getLayerLoc(),
         getCLayer()->activity->data,
         getV(),
         originalLayer->getCLayer()->activity->data);
   return status;
}

int GapLayer::updateState(
      double timef,
      double dt,
      const PVLayerLoc *loc,
      float *A,
      float *V,
      float *checkActive) {
   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int num_neurons = nx * ny * nf;
   int nbatch      = loc->nbatch;
   // No need to update V since GapLayer is a CloneVLayer.
   setActivity_GapLayer(
         nbatch,
         num_neurons,
         A,
         V,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         originalLayer->getLayerLoc()->halo.lt,
         originalLayer->getLayerLoc()->halo.rt,
         originalLayer->getLayerLoc()->halo.dn,
         originalLayer->getLayerLoc()->halo.up,
         checkActive,
         ampSpikelet);
   return PV_SUCCESS;
}

int GapLayer::setActivity() {
   const PVLayerLoc *loc = getLayerLoc();
   return setActivity_GapLayer(
         loc->nbatch,
         getNumNeurons(),
         getCLayer()->activity->data,
         getV(),
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         originalLayer->getLayerLoc()->halo.lt,
         originalLayer->getLayerLoc()->halo.rt,
         originalLayer->getLayerLoc()->halo.dn,
         originalLayer->getLayerLoc()->halo.up,
         getCLayer()->activity->data,
         ampSpikelet);
}

} // end namespace PV
