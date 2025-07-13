/*
 * GPUSystemTestProbe.hpp
 * Author: slundquist
 */

#ifndef GPUSYSTEMTESTPROBE_HPP_
#define GPUSYSTEMTESTPROBE_HPP_
#include "columns/Communicator.hpp"
#include "params/PVParams.hpp"
#include "probes/RequireAllZeroActivityProbe.hpp"

namespace PV {

class GPUSystemTestProbe : public RequireAllZeroActivityProbe {
  public:
   GPUSystemTestProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~GPUSystemTestProbe();

  protected:
   virtual void createProbeCheckStats(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
};

} // namespace PV
#endif
