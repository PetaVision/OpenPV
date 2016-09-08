#ifndef GATESUMPOOLTESTLAYER_HPP_ 
#define GATESUMPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class GateSumPoolTestLayer: public PV::ANNLayer{
public:
   GateSumPoolTestLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
}; // end class GateSumPoolTestLayer


} /* namespace PV */
#endif // GATESUMPOOLTESTLAYER_HPP_
