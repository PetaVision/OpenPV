#ifndef COMPARISONLAYER_HPP_ 
#define COMPARISONLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class ComparisonLayer: public PV::ANNLayer{
public:
   ComparisonLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);
};


} /* namespace PV */
#endif
