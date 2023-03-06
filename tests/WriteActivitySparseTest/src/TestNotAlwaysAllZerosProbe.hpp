#ifndef TESTNOTALWAYSALLZEROSPROBE_HPP_
#define TESTNOTALWAYSALLZEROSPROBE_HPP_

#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class TestNotAlwaysAllZerosProbe : public StatsProbeImmediate {
  public:
   TestNotAlwaysAllZerosProbe(const char *name, PVParams *params, Communicator const *comm);
   bool nonzeroValueHasOccurred() { return mNonzeroValueOccurred; }

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);

   // Member variables
  protected:
   bool mNonzeroValueOccurred = false;
}; // end of class TestNotAlwaysAllZerosProbe

} // namespace PV

#endif // TESTNOTALWAYSALLZEROSPROBE_HPP_
