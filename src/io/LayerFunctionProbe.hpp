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

namespace PV {

class LayerFunctionProbe : public StatsProbe {
public:
   LayerFunctionProbe(HyPerLayer * layer, const char * msg);
   LayerFunctionProbe(const char * filename, HyPerLayer * layer, const char * msg);
   LayerFunctionProbe(HyPerLayer * layer, const char * msg, LayerFunction * F);
   LayerFunctionProbe(const char * filename, HyPerLayer * layer, const char * msg, LayerFunction * F);
   virtual ~LayerFunctionProbe();

   LayerFunction * getFunction() {return function;}
   int setFunction(LayerFunction * f);
   virtual int outputState(double timef);

protected:
   LayerFunctionProbe();
   int initLayerFunctionProbe(const char * filename, HyPerLayer * layer, const char * msg, LayerFunction * F);
   virtual int writeState(double timef, HyPerLayer * l, pvdata_t value);
   LayerFunction * function;
};

}  // end namespace PV

#endif /* LAYERFUNCTIONPROBE_HPP_ */
