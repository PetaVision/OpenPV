#ifndef TESTNOTALWAYSALLZEROSPROBE_HPP_
#define TESTNOTALWAYSALLZEROSPROBE_HPP_

#include <columns/Communicator.hpp>
#include <params/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class TestNotAlwaysAllZerosProbe : public StatsProbeImmediate {
  public:
   TestNotAlwaysAllZerosProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   bool nonzeroValueHasOccurred() { return mNonzeroValueOccurred; }

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   // Member variables
  protected:
   bool mNonzeroValueOccurred = false;
}; // end of class TestNotAlwaysAllZerosProbe

} // namespace PV

#endif // TESTNOTALWAYSALLZEROSPROBE_HPP_
