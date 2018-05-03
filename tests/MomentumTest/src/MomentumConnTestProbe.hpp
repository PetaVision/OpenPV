/*
 * MomentumConnTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef MOMENTUMCONNTESTPROBE_HPP_
#define MOMENTUMCONNTESTPROBE_HPP_

#include "probes/KernelProbe.hpp"

namespace PV {

class MomentumConnTestProbe : public KernelProbe {
  public:
   MomentumConnTestProbe(const char *probename, HyPerCol *hc);
   virtual Response::Status outputState(double timestamp) override;
   virtual void ioParam_isViscosity(enum ParamsIOFlag ioFlag);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   int initialize(const char *probename, HyPerCol *hc);
   int isViscosity;
}; // end class MomentumConnTestProbe

} // end namespace PV

#endif /* PLASTICCONNTESTPROBE_HPP_ */
