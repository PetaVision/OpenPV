#ifndef L1NORMLCAPROBELOCAL_HPP_
#define L1NORMLCAPROBELOCAL_HPP_

#include "io/PVParams.hpp"
#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/L1NormProbeLocal.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class L1NormLCAProbeLocal : public L1NormProbeLocal {
  public:
   L1NormLCAProbeLocal(char const *objName, PVParams *params);
   virtual ~L1NormLCAProbeLocal() {}

  protected:
   L1NormLCAProbeLocal() {}
   void initialize(char const *objName, PVParams *params);
   void warnUnnecessaryParameter(char const *paramName);
};

} // namespace PV

#endif // L1NORMLCAPROBELOCAL_HPP_
