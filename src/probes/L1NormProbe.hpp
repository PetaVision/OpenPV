#ifndef L1NORMPROBE_HPP_
#define L1NORMPROBE_HPP_

#include "columns/Communicator.hpp"
#include "probes/AbstractNormProbe.hpp"
#include "probes/L1NormProbeLocal.hpp"

namespace PV {

class L1NormProbe : public AbstractNormProbe {
  public:
   L1NormProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~L1NormProbe() {}

  protected:
   L1NormProbe() {}

   virtual void createProbeLocal(char const *name, PVParams *params) override;

   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} // namespace PV

#endif // L1NORMPROBE_HPP_
