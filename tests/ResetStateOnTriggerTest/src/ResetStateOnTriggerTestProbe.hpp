#ifndef RESETSTATEONTRIGGERTESTPROBE_HPP_
#define RESETSTATEONTRIGGERTESTPROBE_HPP_

#include "probes/LayerProbe.hpp"

class ResetStateOnTriggerTestProbe : public PV::LayerProbe {
  public:
   ResetStateOnTriggerTestProbe(char const *name, PV::HyPerCol *hc);
   virtual ~ResetStateOnTriggerTestProbe();

   /**
    * Returns zero if the test has passed so far; returns nonzero otherwise.
    */
   int getProbeStatus() { return probeStatus; }

   /**
    * Returns the time of the first failure if the test has failed (i.e. getProbeStatus() returns
    * nonzero)
    * Undefined if the test is still passing.
    */
   double getFirstFailureTime() { return firstFailureTime; }

  protected:
   ResetStateOnTriggerTestProbe();
   int initialize(char const *name, PV::HyPerCol *hc);

   virtual PV::Response::Status
   initializeState(std::shared_ptr<PV::InitializeStateMessage const> message) override;

   /**
    * Returns the number of neurons in the target layer that differ from the expected value.
    */
   void calcValues(double timevalue) override;

   virtual PV::Response::Status outputState(double simTime, double deltaTime) override;

  private:
   int initialize_base();

   // Member variables
  protected:
   double mDeltaTime = 1.0; // Set during InitializeState, and used in calcValues.
   int probeStatus;
   double firstFailureTime;
};

PV::BaseObject *createResetStateOnTriggerTestProbe(char const *name, PV::HyPerCol *hc);

#endif // RESETSTATEONTRIGGERTESTPROBE_HPP_
