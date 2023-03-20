#ifndef ABSTRACTNORMPROBE_HPP_
#define ABSTRACTNORMPROBE_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/EnergyProbeComponent.hpp"
#include "probes/NormProbeAggregator.hpp"
#include "probes/NormProbeLocalInterface.hpp"
#include "probes/NormProbeOutputter.hpp"
#include "probes/ProbeInterface.hpp"
#include "probes/ProbeTriggerComponent.hpp"
#include "probes/TargetLayerComponent.hpp"

#include <memory>

namespace PV {

class AbstractNormProbe : public ProbeInterface {
  public:
   AbstractNormProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~AbstractNormProbe() {}

   HyPerLayer *getTargetLayer() { return mProbeTargetLayer->getTargetLayer(); }
   HyPerLayer const *getTargetLayer() const { return mProbeTargetLayer->getTargetLayer(); }
   char const *getTargetLayerName() const { return mProbeTargetLayer->getTargetLayerName(); }

  protected:
   AbstractNormProbe() {}

   virtual Response::Status allocateDataStructures() override;

   virtual void calcValues(double timestamp) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual void createComponents(char const *name, PVParams *params, Communicator const *comm);

   virtual void createEnergyProbeComponent(char const *name, PVParams *params);
   virtual void createProbeAggregator(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeLocal(char const *name, PVParams *params) = 0;
   virtual void createProbeOutputter(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeTrigger(char const *name, PVParams *params);
   virtual void createTargetLayerComponent(char const *name, PVParams *params);

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void initMessageActionMap() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status outputState(std::shared_ptr<LayerOutputStateMessage const> message);

   virtual Response::Status prepareCheckpointWrite(double simTime) override;
   virtual Response::Status processCheckpointRead(double simTime) override;

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);

  protected:
   // Probe components, set by createComponents(), called by initialize()
   std::shared_ptr<EnergyProbeComponent> mEnergyProbeComponent;
   std::shared_ptr<NormProbeAggregator> mProbeAggregator;
   std::shared_ptr<NormProbeLocalInterface> mProbeLocal;
   std::shared_ptr<NormProbeOutputter> mProbeOutputter;
   std::shared_ptr<TargetLayerComponent> mProbeTargetLayer;
   std::shared_ptr<ProbeTriggerComponent> mProbeTrigger;

  private:
   bool mAddedToEnergyProbe = false; // Set to true when it calls energy probe's addTerm()
   int mLocalNBatch         = 0;
};

} // namespace PV

#endif // ABSTRACTNORMPROBE_HPP_
