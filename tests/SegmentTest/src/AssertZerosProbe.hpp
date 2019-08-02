/*
 * AssertZerosProbe.hpp
 * Author: slundquist
 */

#ifndef ASSERTZEROSPROBE_HPP_
#define ASSERTZEROSPROBE_HPP_
#include "probes/StatsProbe.hpp"

namespace PV {

class AssertZerosProbe : public PV::StatsProbe {
  public:
   AssertZerosProbe(const char *name, PVParams *params, Communicator const *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();

}; // end class AssertZerosProbe

} // end namespace PV
#endif
