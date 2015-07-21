/*
 * AccumulateLayer.hpp
 *
 *  Created on: Nov 18, 2013
 *      Author: pschultz
 *
 *  activity at timestep n+1 is activity(n) + GSyn
 */

#ifndef ACCUMULATELAYER_HPP_
#define ACCUMULATELAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class AccumulateLayer: public PV::ANNLayer {
public:
   AccumulateLayer(const char* name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual ~AccumulateLayer();

protected:
   AccumulateLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_syncedInputLayer(enum ParamsIOFlag ioFlag);
   virtual int setActivity();
   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);

private:
   int initialize_base();

// Member variables
   char * syncedInputLayerName;
   HyPerLayer * syncedInputLayer;
};

} /* namespace PV */
#endif /* ACCUMULATELAYER_HPP_ */
