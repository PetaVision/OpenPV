/*
 * PlasticConnTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef PLASTICCONNTESTPROBE_HPP_
#define PLASTICCONNTESTPROBE_HPP_

#include "probes/KernelProbe.hpp"

namespace PV {

class PlasticConnTestProbe : public KernelProbe {
  public:
   PlasticConnTestProbe(const char *probename, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

   virtual ~PlasticConnTestProbe();

  protected:
   int initialize(const char *probename, HyPerCol *hc);

  protected:
   bool errorPresent;
}; // end class PlasticConnTestProbe

} // end namespace PV

#endif /* PLASTICCONNTESTPROBE_HPP_ */
