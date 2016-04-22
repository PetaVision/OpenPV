/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#ifndef CONSTANTLAYER_HPP_
#define CONSTANTLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class ConstantLayer: public PV::ANNLayer {
public:
   ConstantLayer(const char * name, HyPerCol * hc);
   //virtual int recvAllSynapticInput();
   virtual ~ConstantLayer();
   virtual bool needUpdate(double time, double dt);
protected:
   ConstantLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);
   virtual int communicateInitInfo();
   //virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
   //      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
   //      unsigned int * active_indices, unsigned int * num_active);
   //bool checkIfUpdateNeeded();

private:
   int initialize_base();
}; // class ConstantLayer

BaseObject * createConstantLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
