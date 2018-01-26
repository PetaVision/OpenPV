/*
 * GPUSystemTestProbe.hpp
 * Author: slundquist
 */

#ifndef GPUSYSTEMTESTPROBE_HPP_
#define GPUSYSTEMTESTPROBE_HPP_
#include "probes/StatsProbe.hpp"

namespace PV {

class GPUSystemTestProbe : public PV::StatsProbe {
  public:
   GPUSystemTestProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();
};
}
#endif
