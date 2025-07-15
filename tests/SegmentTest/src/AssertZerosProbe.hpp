/*
 * AssertZerosProbe.hpp
 * Author: slundquist
 */

#ifndef ASSERTZEROSPROBE_HPP_
#define ASSERTZEROSPROBE_HPP_

#include "columns/Communicator.hpp"
#include "params/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class AssertZerosProbe : public PV::StatsProbeImmediate {
  public:
   AssertZerosProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) override;
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

}; // end class AssertZerosProbe

} // end namespace PV
#endif
