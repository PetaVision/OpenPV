/*
 * LayerFunctionProbe.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef LAYERFUNCTIONPROBE_HPP_
#define LAYERFUNCTIONPROBE_HPP_

#include "StatsProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "LayerFunction.hpp"
#include "GenColProbe.hpp"

namespace PV {

class LayerFunctionProbe : public StatsProbe {
public:
   LayerFunctionProbe(const char * probeName, HyPerCol * hc);
   virtual ~LayerFunctionProbe();

   virtual int communicateInitInfo();

   LayerFunction * getFunction() {return function;}
   virtual int outputState(double timef);

protected:
   LayerFunctionProbe();
   int initLayerFunctionProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_parentGenColProbe(enum ParamsIOFlag ioFlag);
   virtual void ioParam_coeff(enum ParamsIOFlag ioFlag);
   virtual void initFunction();
   void setFunction(LayerFunction * f) {function = f;}
   virtual int writeState(double timef, HyPerLayer * l, int batchIdx, pvdata_t value);

private:
   int initLayerFunctionProbe_base();

private:
   LayerFunction * function;
   char * parentGenColProbeName; // The GenColProbe, if any, that the probe is attached to.
   pvdata_t coeff; // The coefficient of this LayerFunction in the energy function of the GenColProbe
};

}  // end namespace PV

#endif /* LAYERFUNCTIONPROBE_HPP_ */
