/*
 * AssertZerosProbe.hpp
 * Author: slundquist
 */

#ifndef ASSERTZEROSPROBE_HPP_ 
#define ASSERTZEROSPROBE_HPP_
#include "probes/StatsProbe.hpp"

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


}  // end namespace PV
#endif 
