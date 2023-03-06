#ifndef STATSPROBE_HPP_
#define STATSPROBE_HPP_

#include "cMakeHeader.h"

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/BaseObject.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "io/PVParams.hpp"
#include "layers/HyPerLayer.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeTriggerComponent.hpp"
#include "probes/StatsProbeAggregator.hpp"
#include "probes/StatsProbeLocal.hpp"
#include "probes/StatsProbeOutputter.hpp"
#include "probes/TargetLayerComponent.hpp"
#include "utils/Timer.hpp"

#include <memory>

namespace PV {

class StatsProbe : public BaseObject {
  protected:
   /**
    * @brief immediateMPIAssembly: If true, assemble stats over MPI each time outputState is
    * called. If false, store the values until a checkpoint, and perform MPI reduction then.
    * The default is false.
    */
   virtual void ioParam_immediateMPIAssembly(enum ParamsIOFlag ioFlag);

  public:
   StatsProbe(char const *name, PVParams *params, Communicator const *comm);
   virtual ~StatsProbe();

   HyPerLayer *getTargetLayer() { return mProbeTargetLayer->getTargetLayer(); }
   HyPerLayer const *getTargetLayer() const { return mProbeTargetLayer->getTargetLayer(); }
   char const *getTargetLayerName() const { return mProbeTargetLayer->getTargetLayerName(); }

  protected:
   StatsProbe();

   void assembleStatsAndOutput();
   virtual void checkStats();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual void createComponents(char const *name, PVParams *params, Communicator const *comm);

   virtual void createProbeAggregator(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeLocal(char const *name, PVParams *params);
   virtual void createProbeOutputter(char const *name, PVParams *params, Communicator const *comm);
   virtual void createProbeTrigger(char const *name, PVParams *params);
   virtual void createTargetLayerComponent(char const *name, PVParams *params);

   void initialize(const char *name, PVParams *params, Communicator const *comm);

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual void initMessageActionMap() override;

   void initProbeTimers(Checkpointer *checkpointer);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status outputState(std::shared_ptr<LayerOutputStateMessage const> message);

   virtual Response::Status prepareCheckpointWrite(double simTime) override;

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   Response::Status respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message);

   bool getImmediateMPIAssembly() const { return mImmediateMPIAssembly; }
   void setImmediateMPIAssembly(bool flag) { mImmediateMPIAssembly = flag; }

  protected:
   // Probe components, set by createComponents(), called by initialize()
   std::shared_ptr<StatsProbeAggregator> mProbeAggregator;
   std::shared_ptr<StatsProbeLocal> mProbeLocal;
   std::shared_ptr<StatsProbeOutputter> mProbeOutputter;
   std::shared_ptr<TargetLayerComponent> mProbeTargetLayer;
   std::shared_ptr<ProbeTriggerComponent> mProbeTrigger;

  private:
   // Private data members
   bool mImmediateMPIAssembly = false;

   Timer *mTimerComp           = nullptr; // A timer for the basic computation of the stats
   Timer *mTimerInitialization = nullptr; // A timer for initialization activity
   Timer *mTimerIO             = nullptr; // A timer for writing output
#ifdef PV_USE_MPI
   Timer *mTimerMPI = nullptr; // A timer for MPI activity
#endif // PV_USE_MPI
};

} // namespace PV

#endif // STATSPROBE_HPP_
