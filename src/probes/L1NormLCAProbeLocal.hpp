#ifndef L1NORMLCAPROBELOCAL_HPP_
#define L1NORMLCAPROBELOCAL_HPP_

#include "probes/CostFunctionSum.hpp"
#include "probes/CostFunctions.hpp"
#include "probes/L1NormProbeLocal.hpp"
#include "probes/NormProbeLocalTemplate.hpp"
#include <memory>

namespace PV {

class L1NormLCAProbeLocal : public L1NormProbeLocal {
  public:
   L1NormLCAProbeLocal(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~L1NormLCAProbeLocal() {}

  protected:
   L1NormLCAProbeLocal() {}
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
};

} // namespace PV

#endif // L1NORMLCAPROBELOCAL_HPP_
