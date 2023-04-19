/*
 * GPUSystemTestProbe.hpp
 * Author: slundquist
 */

#ifndef GPUSYSTEMTESTPROBE_HPP_
#define GPUSYSTEMTESTPROBE_HPP_
#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/RequireAllZeroActivityProbe.hpp"

namespace PV {

class GPUSystemTestProbe : public RequireAllZeroActivityProbe {
  public:
   GPUSystemTestProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~GPUSystemTestProbe();

  protected:
   virtual void createProbeCheckStats(char const *name, PVParams *params) override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(char const *name, PVParams *params, Communicator const *comm);
};

} // namespace PV
#endif
