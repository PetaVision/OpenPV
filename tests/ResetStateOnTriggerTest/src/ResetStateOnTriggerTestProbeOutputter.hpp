#ifndef RESETSTATEONTRIGGERTESTPROBEOUTPUTTER_HPP_
#define RESETSTATEONTRIGGERTESTPROBEOUTPUTTER_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "io/PrintStream.hpp"
#include "probes/BaseProbeOutputter.hpp"
#include "probes/ProbeData.hpp"

#include <memory>

class ResetStateOnTriggerTestProbeOutputter : public PV::BaseProbeOutputter {
  public:
   ResetStateOnTriggerTestProbeOutputter(
         char const *objName,
         PV::PVParams *params,
         PV::Communicator const *comm);
   virtual ~ResetStateOnTriggerTestProbeOutputter() {}

   void printGlobalStatsBuffer(PV::ProbeData<int> const &globalDiscrepancies);

   /**
    * Returns true if printGlobalStatsBuffer() has at some point been given a
    * nonzero discrepancy count; returns false if there has not yet been a discrepancy.
    */
   bool foundDiscrepancies() const { return mDiscrepanciesFound; }

   /**
    * Returns the time of the first failure if the test has failed (i.e. getProbeStatus() returns
    * nonzero)
    * Undefined if the test is still passing.
    */
   double getFirstFailureTime() const { return mFirstFailureTime; }

  protected:
   ResetStateOnTriggerTestProbeOutputter() {}
   void initialize(char const *objName, PV::PVParams *params, PV::Communicator const *comm);

   void printDiscrepancies(
         std::shared_ptr<PV::PrintStream> printStreamPtr,
         int numDiscrepancies,
         double timestamp,
         int batchIndex);

  private:
   bool mDiscrepanciesFound = false;
   double mFirstFailureTime = 0.0;

}; // class ResetStateOnTriggerTestProbeOutputter

#endif // RESETSTATEONTRIGGERTESTPROBEOUTPUTTER_HPP_
