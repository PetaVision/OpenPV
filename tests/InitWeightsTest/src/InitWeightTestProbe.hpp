/*
 * InitWeightTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef INITWEIGHTTESTPROBE_HPP_
#define INITWEIGHTTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class InitWeightTestProbe : public PV::StatsProbeImmediate {
  public:
   InitWeightTestProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} /* namespace PV */
#endif // INITWEIGHTTESTPROBE_HPP_
