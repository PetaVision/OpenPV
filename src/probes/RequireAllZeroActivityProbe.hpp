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

#include "columns/Communicator.hpp"
#include "observerpattern/Response.hpp"
#include "probes/CheckStatsAllZeros.hpp"
#include "probes/StatsProbeImmediate.hpp"

#include <memory>

namespace PV {

class RequireAllZeroActivityProbe : public StatsProbeImmediate {
  public:
   RequireAllZeroActivityProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~RequireAllZeroActivityProbe();

   bool foundNonzero() const { return mCheckStats->foundNonzero(); }
   double getFirstFailureTime() const { return mCheckStats->getFirstFailureTime(); }

  protected:
   RequireAllZeroActivityProbe();

   virtual void checkStats() override;
   virtual Response::Status cleanup() override;
   virtual void
   createComponents(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm) override;
   virtual void createProbeCheckStats(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   // Probe component, set by createComponents(), called by initialize()
   std::shared_ptr<CheckStatsAllZeros> mCheckStats;
}; // end class RequireAllZeroActivityProbe

} /* namespace PV */
#endif /* REQUIREALLZEROACTIVITYPROBE_HPP_ */
