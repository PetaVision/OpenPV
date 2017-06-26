#ifndef TESTNOTALWAYSALLZEROSPROBE_HPP_
#define TESTNOTALWAYSALLZEROSPROBE_HPP_

#include "layers/HyPerLayer.hpp"
#include "probes/StatsProbe.hpp"
#include "utils/PVLog.hpp"

namespace PV {

class TestNotAlwaysAllZerosProbe : public StatsProbe {
  public:
   TestNotAlwaysAllZerosProbe(const char *probeName, HyPerCol *hc);
   bool nonzeroValueHasOccurred() { return nonzeroValueOccurred; }

   virtual int outputState(double timed) override;

  protected:
   int initTestNotAlwaysAllZerosProbe(const char *probeName, HyPerCol *hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initTestNotAlwaysAllZerosProbe_base();

   // Member variables
  protected:
   bool nonzeroValueOccurred;
}; // end of class TestNotAlwaysAllZerosProbe

} // namespace PV

#endif // TESTNOTALWAYSALLZEROSPROBE_HPP_
