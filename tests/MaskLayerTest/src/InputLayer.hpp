#ifndef INPUTLAYER_HPP_ 
#define INPUTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class InputLayer: public PV::ANNLayer{
public:
   InputLayer(const char* name, HyPerCol * hc);
protected:
   virtual int updateState(double timef, double dt);

private:
};


} /* namespace PV */
#endif
