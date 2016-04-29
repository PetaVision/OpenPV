#ifndef TESTALLZEROSPROBE_HPP_
#define TESTALLZEROSPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>
#include <assert.h>

namespace PV {

class TestAllZerosProbe: public StatsProbe {
public:
   TestAllZerosProbe(const char * probeName, HyPerCol * hc);
   
   virtual int outputState(double timed);

protected:
   int initTestAllZerosProbe(const char * probeName, HyPerCol * hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initTestAllZerosProbe_base() {return PV_SUCCESS;}

}; // end class TestAllZerosProbe

BaseObject * createTestAllZerosProbe(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif // TESTALLZEROSPROBE_HPP_
