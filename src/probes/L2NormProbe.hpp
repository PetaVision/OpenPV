#ifndef L2NORMPROBE_HPP_
#define L2NORMPROBE_HPP_

#include "columns/Communicator.hpp"
#include "probes/AbstractNormProbe.hpp"
#include "probes/L2NormProbeLocal.hpp"

namespace PV {

class L2NormProbe : public AbstractNormProbe {
  public:
   L2NormProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~L2NormProbe() {}

  protected:
   L2NormProbe() {}

   virtual void
   createProbeAggregator(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm) override;

   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
};

} // namespace PV

#endif // L2NORMPROBE_HPP_
