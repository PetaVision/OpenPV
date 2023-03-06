#ifndef QUOTIENTPROBE_HPP_
#define QUOTIENTPROBE_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "probes/ProbeInterface.hpp"
#include "probes/ProbeTriggerComponent.hpp"
#include "probes/QuotientProbeOutputter.hpp"
#include <memory>
#include <vector>

namespace PV {

class QuotientProbe : public ProbeInterface {
  protected:
   virtual void ioParam_denominator(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numerator(enum ParamsIOFlag ioFlag);

   /**
    * @brief valueDescription is an obsolete parameter. Use the message parameter instead.
    */
   virtual void ioParam_valueDescription(enum ParamsIOFlag ioFlag);

  public:
   QuotientProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~QuotientProbe() {}

  protected:
   QuotientProbe() {}
   virtual Response::Status allocateDataStructures() override;
   virtual void calcValues(double timestamp) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void createComponents(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeOutputter(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeTrigger(char const *name, PVParams *params);
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status outputState(double simTime, double deltaTime);

   virtual Response::Status prepareCheckpointWrite(double simTime) override;
   virtual Response::Status processCheckpointRead(double simTime) override;

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status
         respondColProbeOutputState(std::shared_ptr<ColProbeOutputStateMessage const>(message));
   Response::Status respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message);

  protected:
   // Probe components, set by createComponents(), called by initialize()
   std::shared_ptr<QuotientProbeOutputter> mProbeOutputter;
   std::shared_ptr<ProbeTriggerComponent> mProbeTrigger;

  private:
   ProbeInterface *mDenominator = nullptr;
   char *mDenominatorName       = nullptr;
   ProbeInterface *mNumerator   = nullptr;
   char *mNumeratorName         = nullptr;
   ProbeDataBuffer<double> mStoredValues;
};

} // namespace PV

#endif // QUOTIENTPROBE_HPP_
