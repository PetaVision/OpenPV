/*
 * CloneVLayer.hpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#ifndef CLONEVLAYER_HPP_
#define CLONEVLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class CloneVLayer: public PV::HyPerLayer {
public:
   CloneVLayer(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int requireChannel(int channelNeeded, int * numChannelsResult);
   virtual int allocateGSyn();
   virtual int requireMarginWidth(int marginWidthNeeded, int * marginWidthResult, char axis);
   virtual bool activityIsSpiking() { return false;}
   HyPerLayer * getOriginalLayer() { return originalLayer; }
   virtual ~CloneVLayer();

protected:
   CloneVLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_originalLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);
   virtual int allocateV();
   virtual int initializeV();
   virtual int readVFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int checkpointWrite(const char * cpDir);
   virtual int updateState(double timed, double dt);

private:
   int initialize_base();

protected:
   char * originalLayerName;
   HyPerLayer * originalLayer;
}; // class CloneVLayer

} /* namespace PV */
#endif /* CLONEVLAYER_HPP_ */
