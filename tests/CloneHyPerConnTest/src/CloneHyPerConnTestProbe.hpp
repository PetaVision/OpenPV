/*
 * CloneHyPerConnTestProbe.hpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONNTESTPROBE_HPP_
#define CLONEKERNELCONNTESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class CloneHyPerConnTestProbe : public PV::StatsProbe {
  public:
   CloneHyPerConnTestProbe(const char *name, PVParams *params, Communicator const *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
