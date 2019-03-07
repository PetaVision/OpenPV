/*
 * identicalBatchProbe.hpp
 * Author: slundquist
 */

#ifndef IDENTICALFEATUREPROBE_HPP_
#define IDENTICALFEATUREPROBE_HPP_
#include "probes/StatsProbe.hpp"

namespace PV {

class identicalBatchProbe : public PV::StatsProbe {
  public:
   identicalBatchProbe(const char *name, PVParams *params, Communicator const *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
};
}
#endif
