/*
 * ComparisonLayer.hpp
 * Author: slundquist
 */

#ifndef COMPARISONLAYER_HPP_ 
#define COMPARISONLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PV{

class ComparisonLayer : public PV::ANNLayer{
public:
   ComparisonLayer(const char * name, HyPerCol * hc);
   virtual int updateState(double timef, double dt);
};

}
#endif 
