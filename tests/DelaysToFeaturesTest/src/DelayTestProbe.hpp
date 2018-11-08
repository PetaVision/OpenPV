/*
 * DelayTestProbe.hpp
 *
 *  Created on: October 1, 2013
 *      Author: wchavez
 */

#ifndef DelayTestProbe_HPP_
#define DelayTestProbe_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class DelayTestProbe : public PV::StatsProbe {
  public:
   DelayTestProbe(const char *name, PVParams *params, Communicator *comm);
   virtual ~DelayTestProbe();

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator *comm);

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* DelayTestProbe_HPP_ */
