#ifndef TESTNOTALWAYSALLZEROSPROBE_HPP_
#define TESTNOTALWAYSALLZEROSPROBE_HPP_

#include "layers/HyPerLayer.hpp"
#include "probes/StatsProbe.hpp"
#include "utils/PVLog.hpp"

namespace PV {

class TestNotAlwaysAllZerosProbe : public StatsProbe {
  public:
   TestNotAlwaysAllZerosProbe(const char *name, PVParams *params, Communicator const *comm);
   bool nonzeroValueHasOccurred() { return nonzeroValueOccurred; }

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();

   // Member variables
  protected:
   bool nonzeroValueOccurred;
}; // end of class TestNotAlwaysAllZerosProbe

} // namespace PV

#endif // TESTNOTALWAYSALLZEROSPROBE_HPP_
