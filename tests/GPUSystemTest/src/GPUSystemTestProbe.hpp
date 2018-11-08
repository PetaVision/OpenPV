/*
 * GPUSystemTestProbe.hpp
 * Author: slundquist
 */

#ifndef GPUSYSTEMTESTPROBE_HPP_
#define GPUSYSTEMTESTPROBE_HPP_
#include "probes/RequireAllZeroActivityProbe.hpp"

namespace PV {

class GPUSystemTestProbe : public PV::RequireAllZeroActivityProbe {
  public:
   GPUSystemTestProbe(const char *name, PVParams *params, Communicator *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator *comm);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();
};
}
#endif
