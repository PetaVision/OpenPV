#ifndef L1NORMPROBE_HPP_
#define L1NORMPROBE_HPP_

#include "columns/Communicator.hpp"
#include "probes/AbstractNormProbe.hpp"
#include "probes/L1NormProbeLocal.hpp"

namespace PV {

class L1NormProbe : public AbstractNormProbe {
  public:
   L1NormProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~L1NormProbe() {}

  protected:
   L1NormProbe() {}

   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
};

} // namespace PV

#endif // L1NORMPROBE_HPP_
