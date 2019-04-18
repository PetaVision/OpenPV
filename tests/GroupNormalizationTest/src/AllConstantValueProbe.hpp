/*
 * AllConstantValueProbe.hpp
 *
 * A probe to check that a layer is constant, with a value given by the parameter "correctValue"
 */

#ifndef ALLCONSTANTVALUEPROBE_HPP_
#define ALLCONSTANTVALUEPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class AllConstantValueProbe : public StatsProbe {
  public:
   AllConstantValueProbe(const char *name, HyPerCol *hc);
   ~AllConstantValueProbe();

   float getCorrectValue() { return correctValue; }

   virtual Response::Status outputState(double timestamp) override;

  protected:
   AllConstantValueProbe();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_correctValue(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

   // Member variables
   float correctValue;
}; // class AllConstantValueProbe

} // namespace PV

#endif // ALLCONSTANTVALUEPROBE_HPP_
