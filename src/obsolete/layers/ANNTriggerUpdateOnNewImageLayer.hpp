/*
 * ANNTriggerUpdateOnNewImageLayer.hpp
 *
 *  Created on: Jul 26, 2013
 *      Author: gkenyon
 */

#ifdef OBSOLETE // Marked obsolete April 23, 2014.
// Use ANNLayer with triggerFlag set to true and triggerLayerName for the triggering layer

#ifndef ANNTRIGGERUPDATEONNEWIMAGELAYER_HPP_
#define ANNTRIGGERUPDATEONNEWIMAGELAYER_HPP_

#include "ANNLayer.hpp"
#include "Movie.hpp"

namespace PV {

class ANNTriggerUpdateOnNewImageLayer: public PV::ANNLayer {
public:
   ANNTriggerUpdateOnNewImageLayer(const char * name, HyPerCol * hc);
   //virtual int recvAllSynapticInput();
   virtual ~ANNTriggerUpdateOnNewImageLayer();
   virtual bool needUpdate(double time, double dt);
protected:
   ANNTriggerUpdateOnNewImageLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_movieLayerName(enum ParamsIOFlag ioFlag);
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

#endif // OBSOLETE
