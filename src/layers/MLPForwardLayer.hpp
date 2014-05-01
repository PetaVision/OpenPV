/*
 * MLPForwardLayer.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#ifndef MLPFORWARDLAYER_HPP_
#define MLPFORWARDLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class MLPForwardLayer: public PV::ANNLayer {
public:
   MLPForwardLayer(const char * name, HyPerCol * hc);
   virtual ~MLPForwardLayer();
   bool * getDropout() {return dropout;}
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
protected:
   MLPForwardLayer();
   virtual int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_DropoutChance(enum ParamsIOFlag ioFlag);
   virtual void ioParam_PotentialScale(enum ParamsIOFlag ioFlag);
   virtual int updateState(double time, double dt);
private:
   int initialize_base();
   bool * dropout;
   float dropoutChance;
   float potentialScale;
   Random * randState;
};

} /* namespace PV */
#endif /* ANNERRORLAYER_HPP_ */
