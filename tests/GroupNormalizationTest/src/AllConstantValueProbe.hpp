/*
 * AllConstantValueProbe.hpp
 *
 * A probe to check that a layer is constant, with a value given by the parameter "correctValue"
 */

#ifndef ALLCONSTANTVALUEPROBE_HPP_
#define ALLCONSTANTVALUEPROBE_HPP_

#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class AllConstantValueProbe : public StatsProbeImmediate {
  public:
   AllConstantValueProbe(const char *name, PVParams *params, Communicator const *comm);
   ~AllConstantValueProbe();

   float getCorrectValue() { return mCorrectValue; }

  protected:
   AllConstantValueProbe();
   virtual void checkStats() override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_correctValue(enum ParamsIOFlag ioFlag);

  private:
   // Member variables
   float mCorrectValue = 0.0f;
}; // class AllConstantValueProbe

} // namespace PV

#endif // ALLCONSTANTVALUEPROBE_HPP_
