#ifndef FIRMTHRESHOLDCOSTFNPROBE_HPP_
#define FIRMTHRESHOLDCOSTFNPROBE_HPP_

#include "columns/Communicator.hpp"
#include "probes/AbstractNormProbe.hpp"
#include "probes/FirmThresholdCostFnProbeLocal.hpp"

namespace PV {

class FirmThresholdCostFnProbe : public AbstractNormProbe {
  public:
   FirmThresholdCostFnProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~FirmThresholdCostFnProbe() {}

  protected:
   FirmThresholdCostFnProbe() {}

   virtual void createProbeLocal(char const *name, PVParams *params) override;

   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} // namespace PV

#endif // FIRMTHRESHOLDCOSTFNPROBE_HPP_
