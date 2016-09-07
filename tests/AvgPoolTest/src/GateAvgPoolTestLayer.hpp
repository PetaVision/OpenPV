#ifndef GATEAVGPOOLTESTLAYER_HPP_
#define GATEAVGPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class GateAvgPoolTestLayer: public PV::ANNLayer{
public:
   GateAvgPoolTestLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
}; // end class GateAvgPoolTestLayer


} /* namespace PV */
#endif // GATEAVGPOOLTESTLAYER_HPP_
