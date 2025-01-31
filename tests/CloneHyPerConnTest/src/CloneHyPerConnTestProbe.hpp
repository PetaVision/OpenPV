/*
 * CloneHyPerConnTestProbe.hpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#ifndef CLONEHYPERCONNTESTPROBE_HPP_
#define CLONEHYPERCONNTESTPROBE_HPP_

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
#endif /* CLONEHYPERCONNTESTPROBE_HPP_ */
