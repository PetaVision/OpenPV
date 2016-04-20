/*
 * AllConstantValueProbe.hpp
 *
 * A probe to check that a layer is constant, with a value given by the parameter "correctValue"
 */

#ifndef ALLCONSTANTVALUEPROBE_HPP_
#define ALLCONSTANTVALUEPROBE_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class AllConstantValueProbe : public StatsProbe {
public:
   AllConstantValueProbe(const char * probeName, HyPerCol * hc);
   ~AllConstantValueProbe();

   pvadata_t getCorrectValue() { return correctValue; }

   int outputState(double timed);

protected:
   AllConstantValueProbe();
   int initAllConstantValueProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_correctValue(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

// Member variables
   pvadata_t correctValue;
}; // class AllConstantValueProbe

BaseObject * createAllConstantValueProbe(char const * probeName, HyPerCol * hc);

}  // namespace PV

#endif // ALLCONSTANTVALUEPROBE_HPP_
