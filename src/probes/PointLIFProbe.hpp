/*
 * PointLIFProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef POINTLIFPROBE_HPP_
#define POINTLIFPROBE_HPP_

#include "PointProbe.hpp"

namespace PV {

class PointLIFProbe : public PointProbe {
  public:
   PointLIFProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   PointLIFProbe();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

   /**
    * Overrides initNumValues to set numValues to 6 (G_E, G_I, G_IB, V, Vth, a)
    */
   virtual void initNumValues() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual void setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message);

   /**
    * Overrides PointProbe::calcValues to report the conductances and threshold V
    * as well as V and
    * A.
    * Note that under MPI, only the root process and the process containing the
    * neuron being probed
    * contain
    * the values.
    */
   virtual void calcValues(double timevalue) override;

   virtual void writeState(double timevalue) override;

  protected:
   double writeTime = 0.0; // time of next output
   double writeStep = 0.0; // output time interval

}; // end class PointLIFProbe
}

#endif /* POINTLIFPROBE_HPP_ */
