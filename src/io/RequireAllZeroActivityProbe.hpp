/*
 * RequireAllZeroActivityProbe.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: pschultz
 *
 *  This probe checks whether the target layer has a nonzero activity.
 *  It is designed to be used with GenericSystemTest-type system tests.
 *
 *  It records whether a nonzero activity is ever found, but it does not
 *  immediately exit with an error at that point.  Instead,
 *  the public method getNonzeroFound() returns the value.  This method
 *  can then be checked after HyPerCol::run() returns and before the HyPerCol
 *  is deleted, e.g. in buildandrun's customexit hook.
 */

#ifndef REQUIREALLZEROACTIVITYPROBE_HPP_
#define REQUIREALLZEROACTIVITYPROBE_HPP_

#include "StatsProbe.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class RequireAllZeroActivityProbe: public PV::StatsProbe {
public:
   RequireAllZeroActivityProbe(const char * probeName, HyPerCol * hc);
   virtual ~RequireAllZeroActivityProbe();
   virtual int outputState(double timed);

   bool getNonzeroFound() { return nonzeroFound; }
   double getNonzeroTime() { return nonzeroTime; }

protected:
   RequireAllZeroActivityProbe();
   int initRequireAllZeroActivityProbe(const char * probeName, HyPerCol * hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

protected:
   bool nonzeroFound;
   double nonzeroTime;
}; // end class RequireAllZeroActivityProbe

BaseObject * createRequireAllZeroActivityProbe(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* REQUIREALLZEROACTIVITYPROBE_HPP_ */
