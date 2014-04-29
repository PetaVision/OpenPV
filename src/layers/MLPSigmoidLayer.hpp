/*
 * MLPSigmoidLayer.hpp
 * can be used to implement Sigmoid junctions
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef MLPSIGMOIDLAYER_HPP_
#define MLPSIGMOIDLAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

// MLPSigmoidLayer can be used to implement Sigmoid junctions between spiking neurons
class MLPSigmoidLayer: public CloneVLayer {
public:
   MLPSigmoidLayer(const char * name, HyPerCol * hc);
   virtual ~MLPSigmoidLayer();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();
protected:
   MLPSigmoidLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_LinAlpha(enum ParamsIOFlag ioFlag);
   /* static */ int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, float linear_alpha, unsigned int * active_indices, unsigned int * num_active);
private:
   int initialize_base();
   float linAlpha;
};

}

#endif /* CLONELAYER_HPP_ */
