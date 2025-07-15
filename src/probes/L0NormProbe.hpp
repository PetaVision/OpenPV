#ifndef L0NORMPROBE_HPP_
#define L0NORMPROBE_HPP_

#include "columns/Communicator.hpp"
#include "probes/AbstractNormProbe.hpp"
#include "probes/L0NormProbeLocal.hpp"

namespace PV {

class L0NormProbe : public AbstractNormProbe {
  public:
   L0NormProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~L0NormProbe() {}

  protected:
   L0NormProbe() {}

   virtual void createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) override;

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
};

} // namespace PV

#endif // L0NORMPROBE_HPP_
