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
   GPUSystemTestProbe(const char *probeName, HyPerCol *hc);

   virtual int outputState(double timed) override;

  protected:
   int initGPUSystemTestProbe(const char *probeName, HyPerCol *hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initGPUSystemTestProbe_base();
};
}
#endif
