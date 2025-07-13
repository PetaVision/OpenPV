#ifndef FIRMTHRESHOLDCOSTFNPROBE_HPP_
#define FIRMTHRESHOLDCOSTFNPROBE_HPP_

#include "columns/Communicator.hpp"
#include "probes/AbstractNormProbe.hpp"
#include "probes/FirmThresholdCostFnProbeLocal.hpp"

namespace PV {

class FirmThresholdCostFnProbe : public AbstractNormProbe {
  public:
   FirmThresholdCostFnProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~FirmThresholdCostFnProbe() {}

  protected:
   FirmThresholdCostFnProbe() {}

   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
};

} // namespace PV

#endif // FIRMTHRESHOLDCOSTFNPROBE_HPP_
