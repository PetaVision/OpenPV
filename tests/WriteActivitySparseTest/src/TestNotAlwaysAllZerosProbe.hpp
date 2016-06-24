#ifndef TESTNOTALWAYSALLZEROSPROBE_HPP_
#define TESTNOTALWAYSALLZEROSPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>
#include <assert.h>

namespace PV {

class TestNotAlwaysAllZerosProbe: public StatsProbe {
public:
   TestNotAlwaysAllZerosProbe(const char * probeName, HyPerCol * hc);
   bool nonzeroValueHasOccurred() { return nonzeroValueOccurred; }
   
   virtual int outputState(double timed);

protected:
   int initTestNotAlwaysAllZerosProbe(const char * probeName, HyPerCol * hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initTestNotAlwaysAllZerosProbe_base();

// Member variables
protected:
   bool nonzeroValueOccurred;
}; // end of class TestNotAlwaysAllZerosProbe

BaseObject * createTestNotAlwaysAllZerosProbe(char const * name, HyPerCol * hc);

} // namespace PV

#endif // TESTNOTALWAYSALLZEROSPROBE_HPP_
