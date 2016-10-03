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
   identicalBatchProbe(const char *probeName, HyPerCol *hc);

   virtual int outputState(double timed);

  protected:
   int initidenticalBatchProbe(const char *probeName, HyPerCol *hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

  private:
   int initidenticalBatchProbe_base();
};
}
#endif
