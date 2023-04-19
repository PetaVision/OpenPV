#ifndef L2NORMPROBE_HPP_
#define L2NORMPROBE_HPP_

#include "columns/Communicator.hpp"
#include "probes/AbstractNormProbe.hpp"
#include "probes/L2NormProbeLocal.hpp"

namespace PV {

class L2NormProbe : public AbstractNormProbe {
  public:
   L2NormProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~L2NormProbe() {}

  protected:
   L2NormProbe() {}

   virtual void
   createProbeAggregator(char const *name, PVParams *params, Communicator const *comm) override;

   virtual void createProbeLocal(char const *name, PVParams *params) override;

   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} // namespace PV

#endif // L2NORMPROBE_HPP_
