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
#include "columns/DataStore.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/Publisher.hpp"
#include "columns/Random.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "initv/BaseInitV.hpp"
#include "io/fileio.hpp"
#include "layers/BaseLayer.hpp"
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

class HyPerLayer : public BaseLayer {

  protected:
   /**
    * List of parameters needed from the HyPerLayer class
    * @name HyPerLayer Parameters
    * @{
    */

   virtual void ioParam_dataType(enum ParamsIOFlag ioFlag);

   /**
    * @brief updateGpu: When compiled using CUDA or OpenCL GPU acceleration, this flag tells whether
    * this layer's updateState method should use the GPU.
    * If PetaVision was compiled without GPU acceleration, it is an error to set this flag to true.
    */
   virtual void ioParam_updateGpu(enum ParamsIOFlag ioFlag);

   /**
    * @brief nxScale: Defines the relationship between the x column size and the layer size.
    * @details Must be 2^n or 1/2^n
    */
   virtual void ioParam_nxScale(enum ParamsIOFlag ioFlag);
   /**
    * @brief nyScale: Defines the relationship between the y column size and the layer size.
    * @details Must be 2^n or 1/2^n
    */
   virtual void ioParam_nyScale(enum ParamsIOFlag ioFlag);
   /**
    * @brief nf: Defines how many features this layer has
    */
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag);

   /**
    * @brief phase: Defines the ordering in which each layer is updated
    */
   virtual void ioParam_phase(enum ParamsIOFlag ioFlag);

   /**
    * @brief mirrorBCflag: If set to true, the margin will mirror the data
    */
   virtual void ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag);

   /**
    * @brief valueBC: If mirrorBC is set to true, Uses the specified value for the margin area
    */
   virtual void ioParam_valueBC(enum ParamsIOFlag ioFlag);

   /**
    * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory
    * set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief initVType: Specifies how to initialize the V buffer.
    * @details Possible choices include
    * - @link ConstantV::ioParamsFillGroup ConstantV@endlink: Sets V to a constant value
    * - @link ZeroV::ioParamsFillGroup ZeroV@endlink: Sets V to zero
    * - @link UniformRandomV::ioParamsFillGroup UniformRandomV@endlink: Sets V with a uniform
    * distribution
    * - @link GaussianRandomV::ioParamsFillGroup GaussianRandomV@endlink: Sets V with a gaussian
    * distribution
    * - @link InitVFromFile::ioparamsFillGroup InitVFromFile@endlink: Sets V to specified pvp file
    *
    * Further parameters are needed depending on initialization type.
    */
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);

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

   /**
    * @brief writeSparseValues: No longer used.
    */
   virtual void ioParam_writeSparseValues(enum ParamsIOFlag ioFlag); // obsolete March 14, 2017.
   /** @} */

  private:
   int initialize_base();

  protected:
   // only subclasses can be constructed directly
   HyPerLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int initClayer();

   virtual int allocateClayerBuffers();
   int setLayerLoc(PVLayerLoc *layerLoc, float nxScale, float nyScale, int nf, int numBatches);
   virtual int allocateBuffers();
   virtual int allocateGSyn();
   void addPublisher();

   /*
    * Allocates a buffer of the given length.  The membrane potential and activity buffer, among
    * others, are created using allocateBuffer.
    * To free a buffer created with this method, call freeBuffer().
    */
   template <typename T>
   int allocateBuffer(T **buf, int bufsize, const char *bufname);

   /**
    * Allocates a restricted buffer (that is, buffer's length is getNumNeuronsAllBatches()).
    */
   int allocateRestrictedBuffer(float **buf, const char *bufname);

   /**
    * Allocates an extended buffer (that is, buffer's length is getNumExtendedAllBatches()).
    */
   int allocateExtendedBuffer(float **buf, const char *bufname);

   int allocateCube();
   virtual int allocateV();
   virtual int allocateActivity();
   virtual int allocatePrevActivity();
   virtual int setInitialValues();

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

   virtual int initializeV();
   virtual int initializeActivity();
   virtual int readStateFromCheckpoint(Checkpointer *checkpointer) override;
   virtual int readActivityFromCheckpoint(Checkpointer *checkpointer);
   virtual int readVFromCheckpoint(Checkpointer *checkpointer);
   virtual int readDelaysFromCheckpoint(Checkpointer *checkpointer);
#ifdef PV_USE_CUDA
   virtual int copyInitialStateToGPU() override;
#endif // PV_USE_CUDA

   // readBufferFile and readDataStoreFromFile were removed Jan 23, 2017.
   // They were only used by checkpointing, which is now handled by the
   // CheckpointEntry class hierarchy.

   void updateNBands(int numCalls);

   virtual int processCheckpointRead() override;

   void calcNumExtended();

   /**
    * Returns true if the trigger behavior is resetStateOnTrigger and the layer was triggered.
    */
   virtual bool needReset(double timed, double dt);

   /**
    * Called instead of updateState when triggerBehavior is "resetStateOnTrigger" and a triggering
    * event occurs.
    * Copies the membrane potential V from triggerResetLayer and then calls setActivity to update A.
    */
   virtual int resetStateOnTrigger();

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
      return clayer->activity->data;
   } // TODO: access to clayer->activity->data should not be public
   virtual double getTimeScale(int batchIdx) { return -1.0; };
   virtual bool activityIsSpiking() { return false; }
   PVDataType getDataType() { return dataType; }
   virtual int respond(std::shared_ptr<BaseMessage const> message) override;

  protected:
   /**
    * The function that calls all ioParam functions
    */
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   static int equalizeMargins(HyPerLayer *layer1, HyPerLayer *layer2);

   int freeClayer();

  public:
   virtual ~HyPerLayer();

   void synchronizeMarginWidth(HyPerLayer *layer);

   // TODO - make protected
   PVLayer *clayer;

   // ************************************************************************************//
   // interface for public methods for controlling HyPerLayer cellular and synaptic dynamics
   // (i.e. methods for receiving synaptic input, updating internal state, publishing output)
   // ************************************************************************************//
   virtual int recvAllSynapticInput(); // Calls recvSynapticInput for each conn and each arborID

   // An updateState wrapper that determines if updateState needs to be called
   int callUpdateState(double simTime, double dt);
   /**
     * A virtual function to determine if callUpdateState method needs to be called
     * Default behavior is dependent on the triggering method.
     * If there is no triggering, always returns true.
     * If there is triggering and the trigger behavior is updateOnlyOnTrigger, returns true only
    * when there is a triggering event.
     * If there is triggering and the trigger behavior is resetStateOnTrigger, returns true only
    * when there is not a trigger event.
     * @param time The current timestep of the run
     * @param dt The current non-adaptive dt of the run
     * @return Returns if the update needs to happen
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

   int respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message);
   int respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message);
   int respondLayerProbeWriteParams(std::shared_ptr<LayerProbeWriteParamsMessage const> message);
   int
   respondLayerClearProgressFlags(std::shared_ptr<LayerClearProgressFlagsMessage const> message);
   int respondLayerRecvSynapticInput(std::shared_ptr<LayerRecvSynapticInputMessage const> message);
   int respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message);
#ifdef PV_USE_CUDA
   int respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message);
#endif // PV_USE_CUDA
   int respondLayerAdvanceDataStore(std::shared_ptr<LayerAdvanceDataStoreMessage const> message);
   int respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message);
   int respondLayerCheckNotANumber(std::shared_ptr<LayerCheckNotANumberMessage const> message);
   // respondLayerUpdateActiveIndices removed Feb 3, 2017. Layers update active indices
   // in response to other messages, when needed.
   int respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   virtual int publish(Communicator *comm, double simTime);
   virtual int resetGSynBuffers(double timef, double dt);
   // ************************************************************************************//

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to
   // next time step
   virtual int waitOnPublish(Communicator *comm);

   virtual int updateAllActiveIndices();
   virtual int updateActiveIndices();
   int resetBuffer(float *buf, int numItems);

   static bool localDimensionsEqual(PVLayerLoc const *loc1, PVLayerLoc const *loc2);
   int mirrorInteriorToBorder(PVLayerCube *cube, PVLayerCube *borderCube);

   virtual int outputState(double timef);
   virtual int writeActivity(double timed);
   virtual int writeActivitySparse(double timed);

   virtual int insertProbe(LayerProbe *probe);
   int outputProbeParams();

   /**
    * Returns true if the MPI exchange for the specified delay has finished;
    * false if it is still in process.
    */
   bool isExchangeFinished(int delay = 0);

   int clearProgressFlags();

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

   int getNumNeurons() { return clayer->numNeurons; }
   int getNumExtended() { return clayer->numExtended; }
   int getNumNeuronsAllBatches() { return clayer->numNeuronsAllBatches; }
   int getNumExtendedAllBatches() { return clayer->numExtendedAllBatches; }

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
   virtual int requireMarginWidth(int marginWidthNeeded, int *marginWidthResult, char axis);
   virtual int requireChannel(int channelNeeded, int *numChannelsResult);

   PVLayer *getCLayer() { return clayer; }
   float *getV() { return clayer->V; } // name query
   int getNumChannels() { return numChannels; }
   float *getChannel(ChannelType ch) { // name query
      return (ch < this->numChannels && ch >= 0) ? GSyn[ch] : NULL;
   }
   virtual float getChannelTimeConst(enum ChannelType channel_type) { return 0.0f; };
   int getXScale() { return clayer->xScale; }
   int getYScale() { return clayer->yScale; }

   bool useMirrorBCs() { return this->mirrorBCflag; }
   float getValueBC() { return this->valueBC; }

   bool getSparseFlag() { return this->sparseLayer; }

   int getPhase() { return this->phase; }

   // implementation of LayerDataInterface interface
   //
   const float *getLayerData(int delay = 0);
   PVLayerLoc const *getLayerLoc() const { return &(clayer->loc); }
   bool isExtended() { return true; }

   double getLastUpdateTime() { return mLastUpdateTime; }
   double getNextUpdateTime() { return mLastUpdateTime + getDeltaUpdateTime(); }

   Publisher *getPublisher() { return publisher; }

  protected:
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
   virtual int setMaxPhase(int *maxPhase);
   virtual int registerData(Checkpointer *checkpointer) override;
   virtual int initializeState() override final;
   // Not overridable since all layers should respond to initializeFromCheckpointFlag the same way.
   // initializeState calls the virtual methods readStateFromCheckpoint() and setInitialValues().

   int openOutputStateFile(Checkpointer *checkpointer);
/* static methods called by updateState({long_argument_list})*/

#ifdef PV_USE_CUDA
   virtual int runUpdateKernel();
   virtual int updateStateGpu(double timef, double dt);
#endif
   virtual int updateState(double timef, double dt);
   virtual int setActivity();
   void freeChannels();

   bool mNeedToPublish = true;

   int numChannels; // number of channels
   float **GSyn; // of dynamic length numChannels
   Publisher *publisher = nullptr;

   float nxScale, nyScale; // Size of layer relative to column
   int numFeatures;
   int xmargin, ymargin;

   bool initializeFromCheckpointFlag = true;
   // If parent HyPerCol sets initializeFromCheckpointDir and this flag is set,
   // the initial state is loaded from the initializeFromCheckpointDir.
   // If the flag is false or the parent's initializeFromCheckpointDir is empty,
   // the initial siate is calculated using setInitialValues().

   int numProbes;
   LayerProbe **probes;

   int phase; // All layers with phase 0 get updated before any with phase 1, etc.
   int numDelayLevels; // The number of timesteps in the datastore ring buffer to store older
   // timesteps for connections with delays

   bool mirrorBCflag; // true when mirror BC are to be applied
   float valueBC; // If mirrorBCflag is false, the value of A to fill extended cells with

   double initialWriteTime; // time of next output
   double writeTime; // time of next output
   double writeStep; // output time interval
   CheckpointableFileStream *mOutputStateStream = nullptr; // activity generated by outputState

   bool sparseLayer; // if true, only nonzero activities are saved; if false, all values are saved.
   // bool writeSparseValues; // removed March 14, 2017
   int writeActivityCalls; // Number of calls to writeActivity (written to nbands in the header of
   // the a%d.pvp file)
   int writeActivitySparseCalls; // Number of calls to writeActivitySparse (written to nbands in the
   // header of the a%d.pvp file)

   int *marginIndices; // indices of neurons in margin
   int numMargin; // number of neurons in margin

   unsigned int rngSeedBase; // The starting seed for rng.  The parent HyPerCol reserves
   // {rngSeedbase, rngSeedbase+1,...rngSeedbase+neededRNGSeeds-1} for use
   // by this layer

   char *initVTypeString   = nullptr;
   BaseInitV *mInitVObject = nullptr;

   HyPerLayer **synchronizedMarginWidthLayers;
   int numSynchronizedMarginWidthLayers;

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

   char *dataTypeString;
   PVDataType dataType;

   double mLastUpdateTime;
   double mLastTriggerTime;

   float **thread_gSyn; // Accumulate buffer for each thread, only used if numThreads > 1
   std::vector<BaseConnection *> recvConns;

   bool mHasReceived = false;
   bool mHasUpdated  = false;

// GPU variables
#ifdef PV_USE_CUDA
  public:
   virtual void syncGpu();
   virtual double addGpuTimers();

   void copyAllGSynToDevice();
   void copyAllGSynFromDevice();
   void copyAllVFromDevice();
   void copyAllActivityFromDevice();
   PVCuda::CudaBuffer *getDeviceV() { return d_V; }
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

   PVCuda::CudaBuffer *d_V;
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
