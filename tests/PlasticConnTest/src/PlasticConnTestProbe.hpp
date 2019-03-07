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
   PlasticConnTestProbe(const char *probename, PVParams *params, Communicator const *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

   virtual ~PlasticConnTestProbe();

  protected:
   void initialize(const char *probename, PVParams *params, Communicator const *comm);

  protected:
   bool errorPresent;
}; // end class PlasticConnTestProbe

} // end namespace PV

#endif /* PLASTICCONNTESTPROBE_HPP_ */
