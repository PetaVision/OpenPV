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

private:
   int initialize_base();
}; // class ConstantLayer

BaseObject * createConstantLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
