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
   PointLIFProbe(const char *name, HyPerCol *hc);

  protected:
   PointLIFProbe();
   int initialize(const char *name, HyPerCol *hc);
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

  private:
   /**
    * Used by calcValues to get the buffer data for the components in the
    * target LIF layer's activity component.
    */
   float const *getBufferData(ObserverTable const *table, char const *label);

  protected:
   double writeTime = 0.0; // time of next output
   double writeStep = 0.0; // output time interval

}; // end class PointLIFProbe
}

#endif /* POINTLIFPROBE_HPP_ */
