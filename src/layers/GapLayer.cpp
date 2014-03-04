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
   if (originalLayerName == NULL) {
      fprintf(stderr, "GapLayer \"%s\" error: originalLayerName must be set.\n", name);
      abort();
   }
   // this->sourceLayerName = strdup(originalLayerName);
   // if (this->sourceLayerName==NULL) {
   //    fprintf(stderr, "GapLayer \"%s\" error: unable to copy originalLayerName \"%s\": %s\n", name, originalLayerName, strerror(errno));
   //    abort();
   // }

   this->clayer->layerType = TypeNonspiking;

   // Moved to readAmpSpikelet
   // ampSpikelet = parent->parameters()->value(name,"ampSpikelet",ampSpikelet);

   return status_init;
}

int GapLayer::setParams(PVParams * params) {
   int status = CloneVLayer::setParams(params);
   readAmpSpikelet(params);
   return status;
}

void GapLayer::readAmpSpikelet(PVParams * params) {
   ampSpikelet = parent->parameters()->value(name,"ampSpikelet",ampSpikelet);
}

int GapLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();

   // Handled by CloneVLayer
   // assert(sourceLayerName);
   // HyPerLayer * hyperlayer = parent->getLayerFromName(sourceLayerName);
   // if (hyperlayer == NULL) {
   //    fprintf(stderr, "GapLayer \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n", name, sourceLayerName);
   //    abort();
   // }
   // sourceLayer = dynamic_cast<LIFGap *>(originalLayer);
   // if (sourceLayer == NULL) {
   //    fprintf(stderr, "GapLayer \"%s\" error: originalLayerName \"%s\" is not a LIFGap or LIFGap-derived class.\n", name, originalLayerName);
   //    abort();
   // }
   // const PVLayerLoc * sourceLoc = sourceLayer->getLayerLoc();
   // const PVLayerLoc * thisLoc = getLayerLoc();
   // if (sourceLoc->nx != thisLoc->nx || sourceLoc->ny != thisLoc->ny || sourceLoc->nf != thisLoc->nf) {
   //    fprintf(stderr, "GapLayer \"%s\" must have the same dimensions as source layer \"%s\".\n", name, sourceLayer->getName());
   //    abort();
   // }

   return status;
}

int GapLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();
   return status;
}

int GapLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), originalLayer->getCLayer()->activity->data);
   if( status == PV_SUCCESS  ) status = updateActiveIndices();
   return status;
}

int GapLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, pvdata_t * checkActive) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_GapLayer();
   setActivity_GapLayer(num_neurons, A, V, nx, ny, nf, loc->nb, originalLayer->getLayerLoc()->nb, checkActive, ampSpikelet);
   return PV_SUCCESS;
}

int GapLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_GapLayer(getNumNeurons(), getCLayer()->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb, originalLayer->getLayerLoc()->nb, getCLayer()->activity->data,ampSpikelet);
}

} // end namespace PV

