/*
 * AssertZerosProbe.hpp
 * Author: slundquist
 */

#ifndef ASSERTZEROSPROBE_HPP_
#define ASSERTZEROSPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class AssertZerosProbe : public PV::StatsProbeImmediate {
  public:
   AssertZerosProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);

}; // end class AssertZerosProbe

} // end namespace PV
#endif
