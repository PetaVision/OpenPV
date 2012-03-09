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
   PVLayer * src_clayer = sourceLayer->getCLayer();
   return updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), sourceLayer->getLayerLoc(), sourceLayer->getSpikingFlag(), src_clayer->numActive, src_clayer->activeIndices);
}

int GapLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, const PVLayerLoc * src_loc, bool src_spiking, unsigned int src_num_active, unsigned int * src_active_indices) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_GapLayer();
   setActivity_GapLayer(num_neurons, A, V, nx, ny, nf, loc->nb, src_loc, src_spiking, src_num_active, src_active_indices);
   // resetGSynBuffers(); // Since GapLayer uses sourceLayer's V, it doesn't seem to use the GSyn channels, so no need to blank them?
   updateActiveIndices();
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

