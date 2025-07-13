/*
 * CloneKernelConnTestProbe.hpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONNTESTPROBE_HPP_
#define CLONEKERNELCONNTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "params/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class CloneKernelConnTestProbe : public PV::StatsProbeImmediate {
  public:
   CloneKernelConnTestProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  protected:
   virtual void checkStats() override;
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
};

} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
