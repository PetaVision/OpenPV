#ifndef INPUTLAYER_HPP_ 
#define INPUTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class InputLayer: public PV::ANNLayer{
public:
	InputLayer(const char* name, HyPerCol * hc);
//   virtual int checkpointRead(const char * cpDir, double* timef);
//   virtual int checkpointWrite(const char * cpDir);
//
protected:
   int updateState(double timef, double dt);

private:
}; // end class InputLayer

BaseObject * createInputLayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif
