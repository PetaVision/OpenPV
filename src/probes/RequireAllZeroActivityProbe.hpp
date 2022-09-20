/*
 * RequireAllZeroActivityProbe.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: pschultz
 *
 * This probe checks whether the target layer has a nonzero activity.
 * It is designed to be used with GenericSystemTest-type system tests.
 *
 * It records whether a nonzero activity is ever found, but it does not immediately exit with an
 * error at that point.  Instead, the public method getNonzeroFound() returns the value.  This
 * method can then be checked after HyPerCol::run() returns and before the HyPerCol is deleted,
 * e.g. in buildandrun's customexit hook. */

#ifndef REQUIREALLZEROACTIVITYPROBE_HPP_
#define REQUIREALLZEROACTIVITYPROBE_HPP_

#include "StatsProbe.hpp"

namespace PV {

class RequireAllZeroActivityProbe : public StatsProbe {
  protected:
   /**
    * List of parameters needed from the RequireAllZeroActivityProbe class
    * @name RequireAllZeroActivityProbe Parameters
    * @{
    */
   /**
    * RequireAllZeroActivityProbe does not read the buffer parameter.
    * It is specific to the Activity buffer.
    */
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   /**
    * @brief exitOnFailure:
    * If true, will error out if a nonzero value is encountered.  Default is true. To control when
    * the error is thrown, see immediateExitOnFailure.  If set to false, the presense of a nonzero
    * value can still be retrieved with the getNonzeroFound() method, and the earliest time at which
    * a nonzero value appears is available through getNonzeroTime().
    */
   virtual void ioParam_exitOnFailure(enum ParamsIOFlag ioFlag);
   /**
    * @brief immediateExitOnFailure:
    * determines when finding a nonzero value causes an exit with an error.  If true, outputState
    * will exit on the timestep a nonzero value is detected.  If false, will not error out until the
    * probe is deleted (which usually happens when the HyPerCol is deleted). Parameter is only read
    * if exitOnFailure is true.  Default is true.
    */
   virtual void ioParam_immediateExitOnFailure(enum ParamsIOFlag ioFlag);
   /** @} */ // End of list of RequireAllZeroActivityProbe parameters.
  public:
   RequireAllZeroActivityProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~RequireAllZeroActivityProbe();

   bool getNonzeroFound() { return mNonzeroFound; }
   double getNonzeroTime() { return mNonzeroTime; }

  protected:
   RequireAllZeroActivityProbe();
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

   virtual Response::Status cleanup() override;

  private:
   int initialize_base();

   void errorMessage(double badTime, std::string const &baseMessage, bool fatalError);

  protected:
   bool mExitOnFailure          = true;
   bool mImmediateExitOnFailure = true;
   bool mNonzeroFound           = false;
   double mNonzeroTime          = 0.0;
}; // end class RequireAllZeroActivityProbe

} /* namespace PV */
#endif /* REQUIREALLZEROACTIVITYPROBE_HPP_ */
