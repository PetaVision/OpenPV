#ifndef AVGPOOLTESTINPUTLAYER_HPP_ 
#define AVGPOOLTESTINPUTLAYER_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class AvgPoolTestInputLayer: public PV::ANNLayer{
public:
	AvgPoolTestInputLayer(const char* name, HyPerCol * hc);
//   virtual int checkpointRead(const char * cpDir, double* timef);
//   virtual int checkpointWrite(const char * cpDir);
//
protected:
   int updateState(double timef, double dt);

private:
}; // end class AvgPoolTestInputLayer


} /* namespace PV */
#endif
