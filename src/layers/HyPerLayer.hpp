/**
 * HyPerLayer.hpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 *
 *  The top of the hierarchy for layer classes.
 *
 */

#ifndef HYPERLAYER_HPP_
#define HYPERLAYER_HPP_

#include "checkpointing/CheckpointableFileStream.hpp"
#include "columns/Communicator.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/Publisher.hpp"
#include "columns/Random.hpp"
#include "components/ActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/InternalStateBuffer.hpp"
#include "components/LayerGeometry.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/PhaseParam.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "io/fileio.hpp"
#include "probes/LayerProbe.hpp"
#include "utils/Timer.hpp"

#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif // PV_USE_OPENMP_THREADS

#ifdef PV_USE_CUDA
#include <arch/cuda/CudaBuffer.hpp>
#include <arch/cuda/CudaKernel.hpp>
#include <arch/cuda/CudaTimer.hpp>
#endif // PV_USE_CUDA

#include <vector>

// default constants
#define HYPERLAYER_FEEDBACK_DELAY 1
#define HYPERLAYER_FEEDFORWARD_DELAY 0

namespace PV {

class PVParams;
class BaseConnection;

typedef enum TriggerBehaviorTypeEnum {
   NO_TRIGGER,
   UPDATEONLY_TRIGGER,
   RESETSTATE_TRIGGER
} TriggerBehaviorType;

class HyPerLayer : public ComponentBasedObject {

  protected:
   /**
    * List of parameters needed from the HyPerLayer class
    * @name HyPerLayer Parameters
    * @{
    */

   // The dataType param was marked obsolete Mar 29, 2018.
   /** @brief dataType: no longer used. */
   virtual void ioParam_dataType(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerFlag: (Deprecated) Specifies if this layer is being triggered
    * @details Defaults to false.
    * This flag is deprecated.  To turn triggering off,
    * set triggerLayer to NULL or the empty string.  It is an error to set this
    * flag to false and triggerLayer to a nonempty string.
    */
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerLayerName: Specifies the name of the layer that this layer triggers off of.
    * If set to NULL or the empty string, the layer does not trigger but updates its state on every
    * timestep.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   // TODO: triggerOffset is measured in units of simulation time, not timesteps.  How does
   // adaptTimeStep affect
   // the triggering time?
   /**
    * @brief triggerOffset: If triggerLayer is set, triggers \<triggerOffset\> timesteps before
    * target trigger
    * @details Defaults to 0
    */
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerBehavior: If triggerLayerName is set, this parameter specifies how the trigger
    * is handled.
    * @details The possible values of triggerBehavior are:
    * - "updateOnlyOnTrigger": updateActivity is called (computing activity buffer from GSyn)
    * only on triggering timesteps.  On other timesteps the layer's state remains unchanged.
    * - "resetStateOnTrigger": On timesteps where the trigger occurs, the membrane potential
    * is copied from the layer specified in triggerResetLayerName and setActivity is called.
    * On nontriggering timesteps, updateActivity is called.
    * For backward compatibility, this parameter defaults to updateOnlyOnTrigger.
    */
   virtual void ioParam_triggerBehavior(enum ParamsIOFlag ioFlag);

   /**
    * @brief triggerResetLayerName: If triggerLayerName is set, this parameter specifies the layer
    * to use for updating
    * the state when the trigger happens.  If set to NULL or the empty string, use triggerLayerName.
    */
   virtual void ioParam_triggerResetLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeStep: Specifies how often to output a pvp file for this layer
    * @details Defaults to every timestep. -1 specifies not to write at all.
    */
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);

   /**
    * @brief initialWriteTime: Specifies the first timestep to start outputing pvp files
    */
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief sparseLayer: Specifies if the layer should be considered sparese for optimization and
    * output
    */
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag);
   /** @} */

  private:
   int initialize_base();

  protected:
   // only subclasses can be constructed directly
   HyPerLayer();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual void initMessageActionMap() override;
   virtual void createComponentTable(char const *description) override;
   virtual LayerGeometry *createLayerGeometry();
   virtual PhaseParam *createPhaseParam();
   virtual BoundaryConditions *createBoundaryConditions();
   virtual LayerInputBuffer *createLayerInput();
   virtual ActivityComponent *createActivityComponent();

   virtual Response::Status setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message);

   void addPublisher();

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual void readDelaysFromCheckpoint(Checkpointer *checkpointer);
#ifdef PV_USE_CUDA
   virtual Response::Status copyInitialStateToGPU() override;
#endif // PV_USE_CUDA

   void updateNBands(int numCalls);

   virtual Response::Status processCheckpointRead() override;

   /**
    * Returns true if the trigger behavior is resetStateOnTrigger and the layer was triggered.
    */
   virtual bool needReset(double timed, double dt);

   /**
    * Called instead of updateActivity when triggerBehavior is "resetStateOnTrigger" and a
    * triggering event occurs.
    * Copies the membrane potential V from triggerResetLayer and then calls setActivity to update A.
    */
   void resetStateOnTrigger(double simTime, double deltaTime);

   /**
    * Returns true if each layer that delivers input to this layer
    * has finished its MPI exchange for its delay; false if any of
    * them has not.
    */
   bool isAllInputReady();

  public:
   HyPerLayer(const char *name, PVParams *params, Communicator *comm);
   virtual double getTimeScale(int batchIdx) { return -1.0; };
   virtual bool activityIsSpiking() { return false; }

  protected:
   /**
    * The function that calls all ioParam functions
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   void freeActivityCube();

  public:
   virtual ~HyPerLayer();

   void synchronizeMarginWidth(HyPerLayer *layer);

   // ************************************************************************************//
   // interface for public methods for controlling HyPerLayer cellular and synaptic dynamics
   // (i.e. methods for receiving synaptic input, updating internal state, publishing output)
   // ************************************************************************************//

   // The method called by respondLayerUpdateState, that determines if resetStateOnTrigger needs
   // to be called, and then calls the ActivityComponent's updateActivity method.
   Response::Status callUpdateState(double simTime, double dt);
   /**
     * A virtual function to determine if the layer will update on a given timestep.
     * Default behavior is dependent on the triggering method.
     * If there is triggering with trigger behavior updateOnlyOnTrigger, returns
     * the trigger layer's needUpdate for the time simTime + triggerOffset.
     * Otherwise, returns true if simTime is LastUpdateTime, LastUpdateTime + getDeltaUpdateTime(),
     * LastUpdateTime + 2*getDeltaUpdateTime(), LastUpdateTime + 3*getDeltaUpdateTime(), etc.
     * @return Returns true an update is needed on that timestep, false otherwise.
     */
   virtual bool needUpdate(double simTime, double dt) const;

   /**
    * A function to return the interval between times when updateActivity is needed.
    */
   double getDeltaUpdateTime() const { return mDeltaUpdateTime; }

   /**
    * A function to return the interval between triggering times.  A negative value means that the
    * layer never triggers
    * (either there is no triggerLayer or the triggerLayer never updates).
    */
   double getDeltaTriggerTime() const;

   Response::Status respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message);
   Response::Status respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message);
   Response::Status
   respondLayerProbeWriteParams(std::shared_ptr<LayerProbeWriteParamsMessage const> message);
   Response::Status
   respondLayerClearProgressFlags(std::shared_ptr<LayerClearProgressFlagsMessage const> message);
   Response::Status
   respondLayerRecvSynapticInput(std::shared_ptr<LayerRecvSynapticInputMessage const> message);
   Response::Status respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message);
#ifdef PV_USE_CUDA
   Response::Status respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message);
#endif // PV_USE_CUDA
   Response::Status
   respondLayerAdvanceDataStore(std::shared_ptr<LayerAdvanceDataStoreMessage const> message);
   Response::Status respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message);
   Response::Status
   respondLayerCheckNotANumber(std::shared_ptr<LayerCheckNotANumberMessage const> message);
   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   virtual int publish(Communicator *comm, double simTime);
   // ************************************************************************************//

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to
   // next time step
   virtual int waitOnPublish(Communicator *comm);

   virtual void updateAllActiveIndices();
   void updateActiveIndices();
   int resetBuffer(float *buf, int numItems);

   virtual Response::Status outputState(double timestamp, double deltaTime);
   virtual int writeActivity(double timed);
   virtual int writeActivitySparse(double timed);

   virtual int insertProbe(LayerProbe *probe);
   Response::Status outputProbeParams();

   /**
    * Returns true if the MPI exchange for the specified delay has finished;
    * false if it is still in process.
    */
   bool isExchangeFinished(int delay = 0);

   int getNumProbes() { return numProbes; }
   LayerProbe *getProbe(int n) { return (n >= 0 && n < numProbes) ? probes[n] : NULL; }

   // Public access functions:

   int getNumNeurons() const { return mLayerGeometry->getNumNeurons(); }
   int getNumExtended() const { return mLayerGeometry->getNumExtended(); }
   int getNumNeuronsAllBatches() const { return mLayerGeometry->getNumNeuronsAllBatches(); }
   int getNumExtendedAllBatches() const { return mLayerGeometry->getNumExtendedAllBatches(); }

   int getNumGlobalNeurons() {
      const PVLayerLoc *loc = getLayerLoc();
      return loc->nxGlobal * loc->nyGlobal * loc->nf;
   }
   int getNumGlobalExtended() {
      const PVLayerLoc *loc = getLayerLoc();
      return (loc->nxGlobal + loc->halo.lt + loc->halo.rt)
             * (loc->nyGlobal + loc->halo.dn + loc->halo.up) * loc->nf;
   }
   int getNumDelayLevels() { return numDelayLevels; }

   int increaseDelayLevels(int neededDelay);

   float const *getV() const {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getBufferData();
   }
   float *getV() {
      return mActivityComponent->getComponentByType<InternalStateBuffer>()->getReadWritePointer();
   }
   int getNumChannels() { return mLayerInput->getNumChannels(); }

   // Eventually, anything that calls one of getXScale, getYScale, or getLayerLoc should retrieve
   // the LayerGeometry component, and these get-methods can be removed from HyPerLayer.
   int getXScale() const { return mLayerGeometry->getXScale(); }
   int getYScale() const { return mLayerGeometry->getYScale(); }
   PVLayerLoc const *getLayerLoc() const { return mLayerGeometry->getLayerLoc(); }

   bool getSparseFlag() { return this->sparseLayer; }

   int getPhase() { return mPhaseParam->getPhase(); }

   // implementation of LayerDataInterface interface
   //
   const float *getLayerData(int delay = 0);

   double getLastUpdateTime() { return mLastUpdateTime; }

   Publisher *getPublisher() { return publisher; }

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual void setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message);

   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * This routine initializes the InternalStateBuffer and ActivityBuffer components. It also sets
    * the LastUpdateTime and LastTriggerTime data members to the DeltaTime argument of the message.
    * (The reason for doing so is that if the layer updates every 10th timestep, it generally
    * should update on timesteps 1, 11, 21, etc.; not timesteps 0, 10, 20, etc.
    * InitializeState is the earliest message that passes the HyPerCol's DeltaTime argument.)
    */
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * A virtual method, called by initializeState() to set the interval between times when
    * updateActivity is needed, if the layer does not have a trigger layer. If the layer does have
    * a trigger layer, this method will not be called and the period is set (during InitializeState)
    * to the that layer's DeltaUpdateTime.
    */
   virtual void setNontriggerDeltaUpdateTime(double dt);

   int openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   /**
    * The function, called by callUpdateState, that updates the ActivityComponent's
    * activity buffer.
    */
   virtual Response::Status updateState(double simTime, double deltaTime);

   bool mNeedToPublish = true;

   Publisher *publisher = nullptr;

   int numProbes;
   LayerProbe **probes;

   LayerGeometry *mLayerGeometry = nullptr;

   // All layers with phase 0 get updated before any with phase 1, etc.
   PhaseParam *mPhaseParam = nullptr;

   BoundaryConditions *mBoundaryConditions = nullptr;

   LayerInputBuffer *mLayerInput = nullptr;

   ActivityComponent *mActivityComponent = nullptr;

   int numDelayLevels; // The number of timesteps in the datastore ring buffer to store older
   // timesteps for connections with delays

   double initialWriteTime                      = 0.0; // time of next output
   double writeTime                             = 0.0; // time of next output
   double writeStep                             = 0.0; // output time interval
   CheckpointableFileStream *mOutputStateStream = nullptr; // activity generated by outputState

   bool sparseLayer; // if true, only nonzero activities are saved; if false, all values are saved.
   int writeActivityCalls; // Number of calls to writeActivity (written to nbands in the header of
   // the a%d.pvp file)
   int writeActivitySparseCalls; // Number of calls to writeActivitySparse (written to nbands in the
   // header of the a%d.pvp file)

   unsigned int rngSeedBase; // The starting seed for rng.  The parent HyPerCol reserves
   // {rngSeedbase, rngSeedbase+1,...rngSeedbase+neededRNGSeeds-1} for use
   // by this layer

   // Trigger-related parameters
   //  Although triggerFlag was deprecated as a params file parameter, it remains as a member
   //  variable to allow quick testing of whether we're triggering.  It is set during
   //  ioParam_triggerLayerName.
   bool triggerFlag; // Whether the layer has different behavior in response to another layer's
   // update.
   char *triggerLayerName; // The layer that triggers different behavior.  To turn triggering off,
   // set this parameter to NULL or ""
   char *triggerBehavior; // Specifies how to respond to a trigger.  Current values are
   // "updateOnlyOnTrigger" or "resetStateOnTrigger"
   TriggerBehaviorType triggerBehaviorType;
   char *triggerResetLayerName; // If triggerBehavior is "resetStateOnTrigger", specifies the layer
   // to use in resetting values.
   double triggerOffset; // Adjust the timestep when the trigger is receieved by this amount; must
   // be >=0.  A positive value means the trigger occurs before the
   // triggerLayerName layer updates.
   HyPerLayer *triggerLayer;
   HyPerLayer *triggerResetLayer;

   double mLastUpdateTime  = 0.0;
   double mLastTriggerTime = 0.0;

   std::vector<BaseConnection *> recvConns;

   bool mHasUpdated = false;

   double mDeltaUpdateTime = 1.0;

// GPU variables
#ifdef PV_USE_CUDA
  public:
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCudnnGSyn() { return cudnn_GSyn; }
#endif // PV_USE_CUDNN
   PVCuda::CudaBuffer *getDeviceDatastore() { return d_Datastore; }

   PVCuda::CudaBuffer *getDeviceActiveIndices() { return d_ActiveIndices; }

   PVCuda::CudaBuffer *getDeviceNumActive() { return d_numActive; }

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCudnnDatastore() { return cudnn_Datastore; }
#endif // PV_USE_CUDNN

   void setAllocDeviceDatastore() { allocDeviceDatastore = true; }

   void setAllocDeviceActiveIndices() { allocDeviceActiveIndices = true; }

   bool getUpdatedDeviceDatastoreFlag() { return updatedDeviceDatastore; }

   void setUpdatedDeviceDatastoreFlag(bool in) { updatedDeviceDatastore = in; }

  protected:
   virtual int allocateDeviceBuffers();
   // OpenCL buffers and their corresponding flags
   //

   bool allocDeviceDatastore;
   bool allocDeviceActiveIndices;
   bool updatedDeviceDatastore;

   PVCuda::CudaBuffer *d_Datastore;
   PVCuda::CudaBuffer *d_numActive;
   PVCuda::CudaBuffer *d_ActiveIndices;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *cudnn_GSyn;
   PVCuda::CudaBuffer *cudnn_Datastore;
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

  protected:
   Timer *publish_timer;
   Timer *timescale_timer;
   Timer *io_timer;
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
