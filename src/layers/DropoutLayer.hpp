#ifndef _DROPOUTLAYER_HPP_
#define _DROPOUTLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class DropoutLayer : public HyPerLayer {

   public:
      DropoutLayer(const char *name, HyPerCol *hc);
      virtual ~DropoutLayer();

   protected:
      virtual int updateState(double timestamp, double dt);  
      virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
      virtual void ioParam_probability(enum ParamsIOFlag ioFlag);

      float mProbability = 0.0f; // Value from 0-1 indicating per-neuron chance of dropout
};

}
#endif
