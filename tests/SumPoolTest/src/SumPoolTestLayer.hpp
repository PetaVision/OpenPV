#ifndef SUMPOOLTESTLAYER_HPP_ 
#define SUMPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class SumPoolTestLayer: public PV::ANNLayer{
public:
   SumPoolTestLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
}; // end class SumPoolTestLayer



} /* namespace PV */
#endif
