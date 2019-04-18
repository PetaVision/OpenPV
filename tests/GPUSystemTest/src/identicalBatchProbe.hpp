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
   identicalBatchProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initidenticalBatchProbe(const char *name, HyPerCol *hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initidenticalBatchProbe_base();
};
}
#endif
