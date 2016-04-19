/*
 * ComparisonLayer.hpp
 * Author: slundquist
 */

#ifndef COMPARISONLAYER_HPP_ 
#define COMPARISONLAYER_HPP_ 
#include <layers/ANNLayer.hpp>

namespace PVMLearning{

class ComparisonLayer : public PV::ANNLayer{
public:
   ComparisonLayer(const char * name, PV::HyPerCol * hc);
   virtual int updateState(double timef, double dt);
}; // end class ComparisonLayer

PV::BaseObject * createComparisonLayer(char const * name, PV::HyPerCol * hc);

}  // end namespace PVMLearning
#endif 
