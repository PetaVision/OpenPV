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
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/CheckStatsAllZeros.hpp"
#include "probes/StatsProbeImmediate.hpp"

#include <memory>

namespace PV {

class RequireAllZeroActivityProbe : public StatsProbeImmediate {
  public:
   RequireAllZeroActivityProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~RequireAllZeroActivityProbe();

   bool foundNonzero() const { return mCheckStats->foundNonzero(); }
   double getFirstFailureTime() const { return mCheckStats->getFirstFailureTime(); }

  protected:
   RequireAllZeroActivityProbe();

   virtual void checkStats() override;
   virtual Response::Status cleanup() override;
   virtual void
   createComponents(char const *name, PVParams *params, Communicator const *comm) override;
   virtual void createProbeCheckStats(char const *name, PVParams *params);
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   // Probe component, set by createComponents(), called by initialize()
   std::shared_ptr<CheckStatsAllZeros> mCheckStats;
}; // end class RequireAllZeroActivityProbe

} /* namespace PV */
#endif /* REQUIREALLZEROACTIVITYPROBE_HPP_ */
