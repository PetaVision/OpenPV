#ifndef L0NORMLCAPROBELOCAL_HPP_
#define L0NORMLCAPROBELOCAL_HPP_

#include "io/PVParams.hpp"
#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/L0NormProbeLocal.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class L0NormLCAProbeLocal : public L0NormProbeLocal {
  protected:
   virtual void ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) override;

  public:
   L0NormLCAProbeLocal(char const *objName, PVParams *params);
   virtual ~L0NormLCAProbeLocal() {}

   void setNnzThreshold(double nnzThreshold);

  protected:
   L0NormLCAProbeLocal() {}
   void initialize(char const *objName, PVParams *params);
   void warnUnnecessaryParameter(char const *paramName);
};

} // namespace PV

#endif // L0NORMLCAPROBELOCAL_HPP_
