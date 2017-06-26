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
   ReceiveFromPostProbe(const char *probeName, HyPerCol *hc);

   virtual int outputState(double timed) override;

  protected:
   int initReceiveFromPostProbe(const char *probeName, HyPerCol *hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   void ioParam_tolerance(enum ParamsIOFlag ioFlag);

  private:
   int initReceiveFromPostProbe_base();

   // Member variables
  protected:
   float tolerance;

}; // end class ReceiveFromPostProbe

} // end namespcae PV

#endif // RECEIVEFROMPOSTPROBE_HPP_
