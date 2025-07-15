#ifndef L0NORMLCAPROBELOCAL_HPP_
#define L0NORMLCAPROBELOCAL_HPP_

#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/L0NormProbeLocal.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class L0NormLCAProbeLocal : public L0NormProbeLocal {
  protected:
   virtual void ioParam_nnzThreshold(ParamsIOSwitch ioSwitch) override;

  public:
   L0NormLCAProbeLocal(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~L0NormLCAProbeLocal() {}

   void setNnzThreshold(double nnzThreshold);

  protected:
   L0NormLCAProbeLocal() {}
   void initialize(std::shared_ptr<ParamsIO> paramsIO);
   void warnUnnecessaryParameter(char const *paramName);
};

} // namespace PV

#endif // L0NORMLCAPROBELOCAL_HPP_
