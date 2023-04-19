/*
 * CloneHyPerConnTestProbe.hpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONNTESTPROBE_HPP_
#define CLONEKERNELCONNTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class CloneHyPerConnTestProbe : public PV::StatsProbeImmediate {
  public:
   CloneHyPerConnTestProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
