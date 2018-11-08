/*
 * ReceiveFromPostProbe.hpp
 * Author: slundquist
 */

#ifndef RECEIVEFROMPOSTPROBE_HPP_
#define RECEIVEFROMPOSTPROBE_HPP_
#include "probes/StatsProbe.hpp"

namespace PV {

class ReceiveFromPostProbe : public PV::StatsProbe {
  public:
   ReceiveFromPostProbe(const char *name, PVParams *params, Communicator *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator *comm);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   void ioParam_tolerance(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

   // Member variables
  protected:
   float tolerance;

}; // end class ReceiveFromPostProbe

} // end namespcae PV

#endif // RECEIVEFROMPOSTPROBE_HPP_
