/*
 * DelayTestProbe.hpp
 *
 *  Created on: October 1, 2013
 *      Author: wchavez
 */

#ifndef DELAYTESTPROBE_HPP_
#define DELAYTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class DelayTestProbe : public PV::StatsProbeImmediate {
  public:
   DelayTestProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~DelayTestProbe();

  protected:
   virtual void checkStats() override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} /* namespace PV */
#endif // DELAYTESTPROBE_HPP_
