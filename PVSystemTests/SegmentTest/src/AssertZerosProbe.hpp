/*
 * AssertZerosProbe.hpp
 * Author: slundquist
 */

#ifndef ASSERTZEROSPROBE_HPP_ 
#define ASSERTZEROSPROBE_HPP_
#include <io/StatsProbe.hpp>

namespace PV{

class AssertZerosProbe : public PV::StatsProbe{
public:
   AssertZerosProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initAssertZerosProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initAssertZerosProbe_base();

}; // end class AssertZerosProbe

BaseObject * createAssertZerosProbe(char const * name, HyPerCol * hc);

}  // end namespace PV
#endif 
