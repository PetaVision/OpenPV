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
   virtual ~ANNTriggerUpdateOnNewImageLayer();
protected:
   ANNTriggerUpdateOnNewImageLayer();
   int initialize(const char * name, HyPerCol * hc, int numChannels, const char * movieLayerName);
   virtual int communicateInitInfo();
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);
private:
   int initialize_base();
   char * movieLayerName;
   Movie * movieLayer;
};

} /* namespace PV */
#endif /* ANNTRIGGERUPDATEONNEWIMAGELAYER_HPP_ */
