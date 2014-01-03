/*
 * ANNTriggerUpdateOnNewImageLayer.cpp
 *
 *  Created on: Jul 26, 2013
 *      Author: gkenyon
 */

//#include "ANNLayer.hpp"
#include "ANNTriggerUpdateOnNewImageLayer.hpp"

namespace PV {

ANNTriggerUpdateOnNewImageLayer::ANNTriggerUpdateOnNewImageLayer()
{
   initialize_base();
}

ANNTriggerUpdateOnNewImageLayer::ANNTriggerUpdateOnNewImageLayer(const char * name, HyPerCol * hc,
		int num_channels, const char * movieLayerName)
{
   initialize_base();
   initialize(name, hc, num_channels, movieLayerName);
}

ANNTriggerUpdateOnNewImageLayer::ANNTriggerUpdateOnNewImageLayer(const char * name, HyPerCol * hc,
		const char * movieLayerName)
{
   initialize_base();
   initialize(name, hc, 2, movieLayerName);
}

ANNTriggerUpdateOnNewImageLayer::~ANNTriggerUpdateOnNewImageLayer()
{
}

int ANNTriggerUpdateOnNewImageLayer::initialize_base()
{
	movieLayerName = NULL;
	movieLayer = NULL;
	return PV_SUCCESS;
}

int ANNTriggerUpdateOnNewImageLayer::initialize(const char * name, HyPerCol * hc,
		int num_channels, const char * movieLayerName)
{
	this->movieLayerName = strdup(movieLayerName);
	if (this->movieLayerName==NULL) {
		fprintf(stderr,
				"ANNTriggerUpdateOnNewImageLayer \"%s\" error: unable to copy movieLayerName \"%s\": %s\n",
				name, this->movieLayerName, strerror(errno));
		exit(EXIT_FAILURE);
	}
	return ANNLayer::initialize(name, hc, num_channels);
}

int ANNTriggerUpdateOnNewImageLayer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();

   HyPerLayer * origHyPerLayer = parent->getLayerFromName(movieLayerName);
   if (origHyPerLayer==NULL) {
      fprintf(stderr, "ANNTriggerUpdateOnNewImageLayer \"%s\" error: movieLayerName \"%s\" is not a layer in the HyPerCol.\n",
    		  name, movieLayerName);
      return(EXIT_FAILURE);
   }
   movieLayer = dynamic_cast<Movie *>(origHyPerLayer);
   if (movieLayer==NULL) {
      fprintf(stderr, "ANNTriggerUpdateOnNewImageLayer \"%s\" error: movieLayerName \"%s\" is not a Movie or Movie-derived layer in the HyPerCol.\n",
    		  name, movieLayerName);
      return(EXIT_FAILURE);
   }

   return status;
}

int ANNTriggerUpdateOnNewImageLayer::recvAllSynapticInput(){
	int status = PV_SUCCESS;
	if (checkIfUpdateNeeded()){
		status = ANNLayer::recvAllSynapticInput();
		// doUpdateState will also need to check movieLayer->getLastUpdateTime() against lastUpdateTime,
		// so wait until then to update lastUpdateTime.
	}
	return status;
}


int ANNTriggerUpdateOnNewImageLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active)
{
   update_timer->start();
   int status = PV_SUCCESS;
   if (checkIfUpdateNeeded()){
	   status = ANNLayer::doUpdateState(time,  dt, loc, A, V, num_channels, gSynHead, spiking, active_indices, num_active);
	   lastUpdateTime = parent->simulationTime();
   }
   update_timer->stop();
   return status;
}

bool ANNTriggerUpdateOnNewImageLayer::checkIfUpdateNeeded() {
   bool needsUpdate = false;
   //Make sure it updates on initialization
   assert(lastUpdateTime >= parent->getStartTime());
   if (lastUpdateTime == parent->getStartTime()){
       return true;
   }
   if (getPhase() > movieLayer->getPhase()) {
      needsUpdate = movieLayer->getLastUpdateTime() >= lastUpdateTime;
   }
   else {
      needsUpdate = movieLayer->getLastUpdateTime() > lastUpdateTime;
   }
   return needsUpdate;
}

} /* namespace PV */

