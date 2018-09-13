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
#include "columns/HyPerCol.hpp"
#include "columns/Publisher.hpp"
#include "columns/Random.hpp"
#include "components/BoundaryConditions.hpp"
#include "components/InitializeFromCheckpointFlag.hpp"
#include "components/InternalStateBuffer.hpp"
#include "components/LayerGeometry.hpp"
#include "components/PhaseParam.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "io/fileio.hpp"
#include "layers/PVLayerCube.hpp"
#include "probes/LayerProbe.hpp"
#include "utils/Timer.hpp"

#ifdef PV_USE_CUDA
#undef PV_USE_CUDA
#include <layers/updateStateFunctions.h>
#define PV_USE_CUDA
#else
#include <layers/updateStateFunctions.h>
#endif // PV_USE_CUDA

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
    * @brief updateGpu: When compiled using CUDA or OpenCL GPU acceleration, this flag tells whether
    * this layer's updateState method should use the GPU.
    * If PetaVision was compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_updateGpu(enum ParamsIOFlag ioFlag);

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
    * - "updateOnlyOnTrigger": updateState is called (computing activity buffer from GSyn)
    * only on triggering timesteps.  On other timesteps the layer's state remains unchanged.
    * - "resetStateOnTrigger": On timesteps where the trigger occurs, the membrane potential
    * is copied from the layer specified in triggerResetLayerName and setActivity is called.
    * On nontriggering timesteps, updateState is called.
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
   int initialize(const char *name, HyPerCol *hc);
   virtual void initMessageActionMap() override;
   virtual void setObserverTable() override;
   virtual LayerGeometry *createLayerGeometry();
   virtual PhaseParam *createPhaseParam();
   virtual BoundaryConditions *createBoundaryConditions();
   virtual InitializeFromCheckpointFlag *createInitializeFromCheckpointFlag();
   virtual InternalStateBuffer *createInternalState();

   virtual Response::Status setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message);

   virtual void allocateBuffers();
   virtual void allocateGSyn();
   void addPublisher();

   /*
    * Allocates a buffer of the given length.  The membrane potential and activity buffer, among
    * others, are created using allocateBuffer.
    * To free a buffer created with this method, call freeBuffer().
    */
   template <typename T>
   void allocateBuffer(T **buf, int bufsize, const char *bufname);

   /**
    * Allocates a restricted buffer (that is, buffer's length is getNumNeuronsAllBatches()).
    */
   void allocateRestrictedBuffer(float **buf, const char *bufname);

   /**
    * Allocates an extended buffer (that is, buffer's length is getNumExtendedAllBatches()).
    */
   void allocateExtendedBuffer(float **buf, const char *bufname);

   void allocateV();
   virtual void allocateActivity();

   void checkpointPvpActivityFloat(
         Checkpointer *checkpointer,
         char const *bufferName,
         float *pvpBuffer,
         bool extended);

   void checkpointRandState(
         Checkpointer *checkpointer,
         char const *bufferName,
         Random *randState,
         bool extendedFlag);

   virtual void initializeActivity();
   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual void readActivityFromCheckpoint(Checkpointer *checkpointer);
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
    * Called instead of updateState when triggerBehavior is "resetStateOnTrigger" and a triggering
    * event occurs.
    * Copies the membrane potential V from triggerResetLayer and then calls setActivity to update A.
    */
   virtual void resetStateOnTrigger();

   /*
    * Frees a buffer created by allocateBuffer().  Note that the address to the buffer
    * is passed as the argument; on return, the address contains NULL.
    * Note that there is no checking whether the buffer was created by allocateBuffer(),
    * or any other allocateBuffer()-related method.
    */
   template <typename T>
   int freeBuffer(T **buf);

   /**
    * Frees a buffer created by allocateRestrictedBuffer().
    * Note that there is no checking whether the buffer was created by allocateRestrictedBuffer(),
    * or any other allocateBuffer()-related method.
    */
   int freeRestrictedBuffer(float **buf);

   /**
    * Frees a buffer created by allocateRestrictedBuffer().
    * Note that there is no checking whether the buffer was created by allocateExtendedBuffer(),
    * or any other allocateBuffer()-related method.
    */
   int freeExtendedBuffer(float **buf);

   /**
    * Returns true if each layer that delivers input to this layer
    * has finished its MPI exchange for its delay; false if any of
    * them has not.
    */
   bool isAllInputReady();

  public:
   HyPerLayer(const char *name, HyPerCol *hc);
   float *getActivity() {
      return mActivityCube->data;
   } // TODO: access to mActivityCube->data should not be public
   virtual double getTimeScale(int batchIdx) { return -1.0; };
   virtual bool activityIsSpiking() { return false; }

  protected:
   /**
    * The function that calls all ioParam functions
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   static int equalizeMargins(HyPerLayer *layer1, HyPerLayer *layer2);

   void freeActivityCube();

  public:
   virtual ~HyPerLayer();

   void synchronizeMarginWidth(HyPerLayer *layer);

   // ************************************************************************************//
   // interface for public methods for controlling HyPerLayer cellular and synaptic dynamics
   // (i.e. methods for receiving synaptic input, updating internal state, publishing output)
   // ************************************************************************************//

   /**
    * Calls deliver for each connection connecting to this layer.
    */
   virtual int recvAllSynapticInput(double simulationTime, double deltaTime);

   // An updateState wrapper that determines if updateState needs to be called
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
   virtual bool needUpdate(double simTime, double dt);

   /**
    * A function to return the interval between times when updateState is needed.
    */
   virtual double getDeltaUpdateTime();

   /**
    * A function to return the interval between triggering times.  A negative value means that the
    * layer never triggers
    * (either there is no triggerLayer or the triggerLayer never updates).
    */
   virtual double getDeltaTriggerTime();

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
   virtual int resetGSynBuffers(double timef, double dt);
   // ************************************************************************************//

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to
   // next time step
   virtual int waitOnPublish(Communicator *comm);

   virtual void updateAllActiveIndices();
   void updateActiveIndices();
   int resetBuffer(float *buf, int numItems);

   static bool localDimensionsEqual(PVLayerLoc const *loc1, PVLayerLoc const *loc2);
   int mirrorInteriorToBorder(PVLayerCube *cube, PVLayerCube *borderCube);

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

   void clearProgressFlags();

   int getNumProbes() { return numProbes; }
   LayerProbe *getProbe(int n) { return (n >= 0 && n < numProbes) ? probes[n] : NULL; }

   // TODO: should the mirroring functions be static?  Why are they virtual?
   virtual int mirrorToNorthWest(PVLayerCube *dest, PVLayerCube *src);
   virtual int mirrorToNorth(PVLayerCube *dest, PVLayerCube *src);
   virtual int mirrorToNorthEast(PVLayerCube *dest, PVLayerCube *src);
   virtual int mirrorToWest(PVLayerCube *dest, PVLayerCube *src);
   virtual int mirrorToEast(PVLayerCube *dest, PVLayerCube *src);
   virtual int mirrorToSouthWest(PVLayerCube *dest, PVLayerCube *src);
   virtual int mirrorToSouth(PVLayerCube *dest, PVLayerCube *src);
   virtual int mirrorToSouthEast(PVLayerCube *dest, PVLayerCube *src);

   /**
    * Adds the given connection to the vector of connections to receive input from.
    * The connection's post-synaptic layer must be the layer for which this
    * member function is called.
    */
   void addRecvConn(BaseConnection *conn);

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
   void requireMarginWidth(int marginWidthNeeded, int *marginWidthResult, char axis);
   virtual int requireChannel(int channelNeeded, int *numChannelsResult);

   PVLayerCube *getActivityCube() { return mActivityCube; }
   float *getV() { return mInternalState->getV(); } // TODO: should be const
   int getNumChannels() { return numChannels; }
   float *getChannel(ChannelType ch) { // name query
      return (ch < this->numChannels && ch >= 0) ? GSyn[ch] : NULL;
   }
   virtual float getChannelTimeConst(enum ChannelType channel_type) { return 0.0f; }

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

   int openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);
/* static methods called by updateState({long_argument_list})*/

#ifdef PV_USE_CUDA
   virtual int runUpdateKernel();
   virtual Response::Status updateStateGpu(double timef, double dt);
#endif
   virtual Response::Status updateState(double timef, double dt);
   virtual int setActivity();
   void freeChannels();

   bool mNeedToPublish = true;

   int numChannels; // number of channels
   float **GSyn; // of dynamic length numChannels
   Publisher *publisher = nullptr;

   bool mInitializeFromCheckpointFlag = false;
   // If parent HyPerCol sets initializeFromCheckpointDir and this flag is set,
   // the initial state is loaded from the initializeFromCheckpointDir.
   // If the flag is false or the parent's initializeFromCheckpointDir is empty,
   // the initial siate is calculated using setInitialValues().

   int numProbes;
   LayerProbe **probes;

   LayerGeometry *mLayerGeometry = nullptr;

   // All layers with phase 0 get updated before any with phase 1, etc.
   PhaseParam *mPhaseParam = nullptr;

   BoundaryConditions *mBoundaryConditions = nullptr;

   InternalStateBuffer *mInternalState = nullptr;

   PVLayerCube *mActivityCube = nullptr;

   int numDelayLevels; // The number of timesteps in the datastore ring buffer to store older
   // timesteps for connections with delays

   double initialWriteTime; // time of next output
   double writeTime; // time of next output
   double writeStep; // output time interval
   CheckpointableFileStream *mOutputStateStream = nullptr; // activity generated by outputState

   bool sparseLayer; // if true, only nonzero activities are saved; if false, all values are saved.
   int writeActivityCalls; // Number of calls to writeActivity (written to nbands in the header of
   // the a%d.pvp file)
   int writeActivitySparseCalls; // Number of calls to writeActivitySparse (written to nbands in the
   // header of the a%d.pvp file)

   int *marginIndices; // indices of neurons in margin
   int numMargin; // number of neurons in margin

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

   bool mHasReceived = false;
   bool mHasUpdated  = false;

// GPU variables
#ifdef PV_USE_CUDA
  public:
   virtual void syncGpu();

   /**
    * Accumulates the times of the gpu_recvsyn_timer and gpu_update_timer.
    * Returns the sum of the total elapsed times on those timers.
    */
   virtual double addGpuTimers();

   void copyAllGSynToDevice();
   void copyAllGSynFromDevice();
   void copyAllActivityFromDevice();
   PVCuda::CudaBuffer *getDeviceGSyn() { return d_GSyn; }

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCudnnGSyn() { return cudnn_GSyn; }
#endif // PV_USE_CUDNN
   PVCuda::CudaBuffer *getDeviceActivity() { return d_Activity; }

   PVCuda::CudaBuffer *getDeviceDatastore() { return d_Datastore; }

   PVCuda::CudaBuffer *getDeviceActiveIndices() { return d_ActiveIndices; }

   PVCuda::CudaBuffer *getDeviceNumActive() { return d_numActive; }

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *getCudnnDatastore() { return cudnn_Datastore; }
#endif // PV_USE_CUDNN

   void setAllocDeviceV() { allocDeviceV = true; }
   void setAllocDeviceGSyn() { allocDeviceGSyn = true; }

   void setAllocDeviceActivity() { allocDeviceActivity = true; }

   void setAllocDeviceDatastore() { allocDeviceDatastore = true; }

   void setAllocDeviceActiveIndices() { allocDeviceActiveIndices = true; }

   bool getUpdatedDeviceActivityFlag() { return updatedDeviceActivity; }

   void setUpdatedDeviceActivityFlag(bool in) { updatedDeviceActivity = in; }

   bool getUpdatedDeviceDatastoreFlag() { return updatedDeviceDatastore; }

   void setUpdatedDeviceDatastoreFlag(bool in) { updatedDeviceDatastore = in; }

   bool getUpdatedDeviceGSynFlag() { return updatedDeviceGSyn; }

   void setUpdatedDeviceGSynFlag(bool in) { updatedDeviceGSyn = in; }

  protected:
   virtual int allocateUpdateKernel();
   virtual int allocateDeviceBuffers();
   // OpenCL buffers and their corresponding flags
   //

   bool allocDeviceV;
   bool allocDeviceGSyn; // array of channels to allocate
   bool allocDeviceActivity;
   bool allocDeviceDatastore;
   bool allocDeviceActiveIndices;
   bool updatedDeviceActivity;
   bool updatedDeviceDatastore;
   bool updatedDeviceGSyn;
   bool mRecvGpu;
   bool mUpdateGpu;

   PVCuda::CudaBuffer *d_GSyn;
   PVCuda::CudaBuffer *d_Activity;
   PVCuda::CudaBuffer *d_Datastore;
   PVCuda::CudaBuffer *d_numActive;
   PVCuda::CudaBuffer *d_ActiveIndices;
   PVCuda::CudaKernel *krUpdate;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *cudnn_GSyn;
   PVCuda::CudaBuffer *cudnn_Datastore;
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

  protected:
   Timer *update_timer;
   Timer *recvsyn_timer;
   Timer *publish_timer;
   Timer *timescale_timer;
   Timer *io_timer;

#ifdef PV_USE_CUDA
   PVCuda::CudaTimer *gpu_recvsyn_timer;
   PVCuda::CudaTimer *gpu_update_timer;
#endif
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
