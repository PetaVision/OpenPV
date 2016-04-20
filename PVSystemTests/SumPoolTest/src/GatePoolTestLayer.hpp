#ifndef GATEPOOLTESTLAYER_HPP_ 
#define GATEPOOLTESTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class GatePoolTestLayer: public PV::ANNLayer{
public:
	GatePoolTestLayer(const char* name, HyPerCol * hc);
   //virtual int checkpointRead(const char * cpDir, double* timef);
   //virtual int checkpointWrite(const char * cpDir);

protected:
   int updateState(double timef, double dt);

private:
}; // end class GatePoolTestLayer

BaseObject * createGatePoolTestLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
