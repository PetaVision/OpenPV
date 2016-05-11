/*
 * GapLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
#include "GapLayer.hpp"

// GapLayer can be used to implement gap junctions
namespace PV {
GapLayer::GapLayer() {
   initialize_base();
}

GapLayer::GapLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

GapLayer::~GapLayer()
{
   // Handled by CloneVLayer constructor
   // clayer->V = NULL;
}

int GapLayer::initialize_base() {
   // sourceLayer = NULL; // Handled by CloneVLayer
   ampSpikelet = 50;
   return PV_SUCCESS;
}

int GapLayer::initialize(const char * name, HyPerCol * hc)
{
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
   parent->ioParamValue(ioFlag, name, "ampSpikelet", &ampSpikelet, ampSpikelet);
}

int GapLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();

   // Handled by CloneVLayer

   return status;
}

int GapLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();
   return status;
}

int GapLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), originalLayer->getCLayer()->activity->data);
   return status;
}

int GapLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, pvdata_t * checkActive) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;
   updateV_GapLayer();
   setActivity_GapLayer(nbatch, num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, originalLayer->getLayerLoc()->halo.lt, originalLayer->getLayerLoc()->halo.rt, originalLayer->getLayerLoc()->halo.dn, originalLayer->getLayerLoc()->halo.up, checkActive, ampSpikelet);
   return PV_SUCCESS;
}

int GapLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_GapLayer(loc->nbatch, getNumNeurons(), getCLayer()->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, originalLayer->getLayerLoc()->halo.lt, originalLayer->getLayerLoc()->halo.rt, originalLayer->getLayerLoc()->halo.dn, originalLayer->getLayerLoc()->halo.up, getCLayer()->activity->data,ampSpikelet);
}

BaseObject * createGapLayer(char const * name, HyPerCol * hc) {
   return hc ? new GapLayer(name, hc) : NULL;
}

} // end namespace PV

