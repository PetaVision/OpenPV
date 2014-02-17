#ifndef TESTALLZEROSPROBE_HPP_
#define TESTALLZEROSPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>
#include <assert.h>

namespace PV {

class TestAllZerosProbe: public StatsProbe {
public:
   TestAllZerosProbe(const char * filename, HyPerLayer * layer, const char * msg);
   TestAllZerosProbe(HyPerLayer * layer, const char * msg);
   
   virtual int outputState(double timed);

protected:
   int initTestAllZerosProbe_base() {return PV_SUCCESS;}
   int initTestAllZerosProbe(const char * filename, HyPerLayer * layer, const char * msg);
};

} // namespace PV

#endif // TESTALLZEROSPROBE_HPP_