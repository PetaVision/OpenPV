#ifndef COLUMNENERGYPROBE_HPP_
#define COLUMNENERGYPROBE_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ColumnEnergyOutputter.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "probes/ProbeInterface.hpp"
#include "probes/ProbeTriggerComponent.hpp"
#include <memory>
#include <vector>

namespace PV {

class ColumnEnergyProbe : public ProbeInterface {
  public:
   ColumnEnergyProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~ColumnEnergyProbe() {}

   /** @brief Adds a ProbeInterface-derived probe to the energy calculation.
    * @details If probe is nullptr, the list of terms is unchanged.
    * The ColumnEnergyProbe class does not try to prevent the same probe
    * from being added more than once.
    * All probes added to the ColumnEnergyProbe must have the same getNumValues().
    * (although this is not checked until registerData() is called, because
    * a probe is typically added during the CommunicateInitInfo stage, but NumValues
    * is typically set during the AllocateDataStructures stage).
    */
   void addTerm(ProbeInterface *probe);

  protected:
   ColumnEnergyProbe() {}
   virtual Response::Status allocateDataStructures() override;
   virtual void calcValues(double timestamp) override;
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
   std::shared_ptr<ColumnEnergyOutputter> mProbeOutputter;
   std::shared_ptr<ProbeTriggerComponent> mProbeTrigger;

  private:
   ProbeDataBuffer<double> mStoredValues;
   std::vector<ProbeInterface *> mTerms;
};

} // namespace PV

#endif // COLUMNENERGYPROBE_HPP_
