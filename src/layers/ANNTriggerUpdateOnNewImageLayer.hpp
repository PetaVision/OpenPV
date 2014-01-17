/*
 * ANNTriggerUpdateOnNewImageLayer.hpp
 *
 *  Created on: Jul 26, 2013
 *      Author: gkenyon
 */

#ifndef ANNTRIGGERUPDATEONNEWIMAGELAYER_HPP_
#define ANNTRIGGERUPDATEONNEWIMAGELAYER_HPP_

#include "ANNLayer.hpp"
#include "Movie.hpp"

namespace PV {

class ANNTriggerUpdateOnNewImageLayer: public PV::ANNLayer {
public:
	ANNTriggerUpdateOnNewImageLayer(const char * name, HyPerCol * hc, int numChannels, const char * movieLayerName);
	ANNTriggerUpdateOnNewImageLayer(const char * name, HyPerCol * hc, const char * movieLayerName);
	//virtual int recvAllSynapticInput();
	virtual ~ANNTriggerUpdateOnNewImageLayer();
#ifdef OBSOLETE
   //Obsolete Jan 15th, 2014 by slundquist
   //getLastUpdateTime in HyPerLayer no loger updates lastUpdateTime, so no longer need to overwrite
	virtual double getLastUpdateTime() {return lastUpdateTime;}
#endif //OBSOLETE
   virtual bool needUpdate(double time, double dt);
protected:
   ANNTriggerUpdateOnNewImageLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels, const char * movieLayerName);
   virtual int communicateInitInfo();
   //virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
   //      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
   //      unsigned int * active_indices, unsigned int * num_active);
   //bool checkIfUpdateNeeded();

private:
   int initialize_base();
   char * movieLayerName;
   Movie * movieLayer;
};

} /* namespace PV */
#endif /* ANNTRIGGERUPDATEONNEWIMAGELAYER_HPP_ */
