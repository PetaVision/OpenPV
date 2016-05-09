#ifndef MAXPOOLTESTLAYER_HPP_ 
#define MAXPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class TestLayer: public PV::ANNLayer{
public:
	TestLayer(const char* name, HyPerCol * hc);

protected:
   int updateState(double timef, double dt);

private:
};

BaseObject * createTestLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
