#ifndef MAXPOOLTESTLAYER_HPP_ 
#define MAXPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class MaxPoolTestLayer: public PV::ANNLayer{
public:
	MaxPoolTestLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
};

BaseObject * createMaxPoolTestLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
