/*
 * CloneLayer.cpp
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

GapLayer::GapLayer(const char * name, HyPerCol * hc, LIFGap * originalLayer) {
   initialize(name, hc, originalLayer);
}

GapLayer::~GapLayer()
{
    clayer->V = NULL;
}

int GapLayer::initialize_base() {
   sourceLayer = NULL;
   return PV_SUCCESS;
}

int GapLayer::initialize(const char * name, HyPerCol * hc, LIFGap * originalLayer)
{
   int status_init = HyPerLayer::initialize(name, hc, MAX_CHANNELS);
   if (originalLayer == NULL) {
      fprintf(stderr, "GapLayer \"%s\" received null source layer.\n", name);
      abort();
   }
   const PVLayerLoc * sourceLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * thisLoc = getLayerLoc();
   if (sourceLoc->nx != thisLoc->nx || sourceLoc->ny != thisLoc->ny || sourceLoc->nf != thisLoc->nf || sourceLoc->nb != thisLoc->nb) {
      fprintf(stderr, "GapLayer \"%s\" must have the same dimensions as source layer \"%s\" (including marginWidth).\n", name, originalLayer->getName());
      abort();
   }
   this->clayer->layerType = TypeNonspiking;
   this->spikingFlag = false;
   sourceLayer = originalLayer;
   free(clayer->V);
   clayer->V = sourceLayer->getV();

   const PVLayerLoc * loc = getLayerLoc();
   // HyPerLayer::setActivity();
   setActivity_HyPerLayer(getNumNeurons(), getCLayer()->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb);
   // this copies the potential into the activity buffer for t=0


   return status_init;
}

int GapLayer::updateState(float timef, float dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), sourceLayer->getCLayer()->activity->data);
   if( status == PV_SUCCESS  ) status = updateActiveIndices();
   return status;
}

int GapLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, pvdata_t * checkActive) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_GapLayer();
   setActivity_GapLayer(num_neurons, A, V, nx, ny, nf, loc->nb, checkActive);
   // resetGSynBuffers(); // Since GapLayer uses sourceLayer's V, it doesn't seem to use the GSyn channels, so no need to blank them?
   return PV_SUCCESS;
}

//// use LIFGap as source layer instead (LIFGap updates gap junctions more accurately)
//int GapLayer::updateV() {
//#ifdef OBSOLETE
//   pvdata_t * V = getV();
//   pvdata_t * GSynExc = getChannel(CHANNEL_EXC);
//   pvdata_t exp_deltaT = 1.0f - exp(-this->getParent()->getDeltaTime() / sourceLayer->getLIFParams()->tau);
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      V[k] += GSynExc[k] * exp_deltaT;  //!!! uses base tau, not the true time-dep tau
//#endif
//   return PV_SUCCESS;
//}

////!!!TODO: add param in LIFGap for spikelet amplitude
int GapLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_GapLayer(getNumNeurons(), getCLayer()->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb, getCLayer()->activity->data);
}

//int GapLayer::setActivity() {
//
//   HyPerLayer::setActivity(); // this copies the potential into the activity buffer
//
//   // extended activity may not be current but this is alright since only local activity is used
//   // !!! will break (non-deterministic) if layers are updated simultaneously--fix is to use datastore
//   const PVLayerLoc * loc = sourceLayer->getLayerLoc();
//   if (sourceLayer->getSpikingFlag()) { // probably not needed since numActive will be zero for non-spiking
//      //pvdata_t * sourceActivity = sourceLayer->getCLayer()->activity->data;
//      pvdata_t * localActivity = getCLayer()->activity->data;
//      unsigned int * activeNdx = sourceLayer->getCLayer()->activeIndices;
//      for (unsigned int kActive = 0; kActive < sourceLayer->getCLayer()->numActive; kActive++) {
//         int kGlobalRestricted = activeNdx[kActive];
//         int kLocalRestricted = localIndexFromGlobal(kGlobalRestricted, *loc);
//         int kLocalExtended = kIndexExtended( kLocalRestricted, loc->nx, loc->ny, loc->nf, loc->nb);
//         localActivity[kLocalExtended] += 50; // add 50 mV spike to local membrane potential
//      }
//   }
//   return PV_SUCCESS;
//}


} // end namespace PV

