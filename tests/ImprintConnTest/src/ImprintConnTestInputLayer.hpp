#ifndef IMPRINTCONNTESTINPUTLAYER_HPP_ 
#define IMPRINTCONNTESTINPUTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class ImprintConnTestInputLayer: public PV::ANNLayer{
public:
	ImprintConnTestInputLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
};


} /* namespace PV */
#endif
