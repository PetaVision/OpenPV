/*
 * CloneKernelConnTestProbe.hpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONNTESTPROBE_HPP_
#define CLONEKERNELCONNTESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class CloneKernelConnTestProbe : public PV::StatsProbe {
  public:
   CloneKernelConnTestProbe(const char *probeName, HyPerCol *hc);

   virtual int outputState(double timed);

  protected:
   int initCloneKernelConnTestProbe(const char *probeName, HyPerCol *hc);

  private:
   int initCloneKernelConnTestProbe_base();
};

} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
