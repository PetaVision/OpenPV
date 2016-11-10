#ifndef _DROPOUTLAYER_HPP_
#define _DROPOUTLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class DropoutLayer : public ANNLayer {

   public:
      DropoutLayer(const char *name, HyPerCol *hc);
      virtual ~DropoutLayer();

   protected:
      virtual int updateState(double timestamp, double dt);  
      virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
      virtual void ioParam_probability(enum ParamsIOFlag ioFlag);

      int mProbability = 0; // Value from 0-99 indicating per-neuron chance of dropout
};

}
#endif
