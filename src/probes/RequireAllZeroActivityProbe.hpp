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

#include "../columns/HyPerCol.hpp"
#include "StatsProbe.hpp"

namespace PV {

class RequireAllZeroActivityProbe : public PV::StatsProbe {
  public:
   RequireAllZeroActivityProbe(const char *name, HyPerCol *hc);
   virtual ~RequireAllZeroActivityProbe();
   virtual Response::Status outputState(double timed) override;

   bool getNonzeroFound() { return nonzeroFound; }
   double getNonzeroTime() { return nonzeroTime; }

  protected:
   RequireAllZeroActivityProbe();
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

   /**
    * List of parameters needed from the RequireAllZeroActivityProbe class
    * @name RequireAllZeroActivityProbe Parameters
    * @{
    */
   /**
    * @brief exitOnFailure: If true, will error out if a nonzero value is
    * encountered.  Default is
    * true.
    * To control when the error is thrown, see immediateExitOnFailure.
    * If set to false, the presense of a nonzero value can still be retrieved
    * with the
    * getNonzeroFound() method,
    * and the earliest time at which a nonzero value appears is available through
    * getNonzeroTime().
    */
   virtual void ioParam_exitOnFailure(enum ParamsIOFlag ioFlag);
   /**
    * @brief immediateExitOnFailure: determines when finding a nonzero value
    * causes an exit with an
    * error.
    * If true, outputState will exit on the timestep a nonzero value is detected.
    * If false,
    * will not error out until the probe is deleted (which usually happens when
    * the HyPerCol is
    * deleted).
    * Parameter is only read if exitOnFailure is true.  Default is true.
    */
   virtual void ioParam_immediateExitOnFailure(enum ParamsIOFlag ioFlag);
   /** @} */ // End of list of RequireAllZeroActivityProbe parameters.

  private:
   int initialize_base();

   void nonzeroFoundMessage(double badTime, bool isRoot, bool fatalError);

  protected:
   bool nonzeroFound           = false;
   bool exitOnFailure          = true;
   bool immediateExitOnFailure = true;
   double nonzeroTime          = 0.0;
}; // end class RequireAllZeroActivityProbe

} /* namespace PV */
#endif /* REQUIREALLZEROACTIVITYPROBE_HPP_ */
