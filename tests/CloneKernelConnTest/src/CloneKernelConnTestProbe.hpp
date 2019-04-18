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
   CloneKernelConnTestProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
