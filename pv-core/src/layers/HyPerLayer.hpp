/**
 * HyPerLayer.hpp
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 *
 *  The top of the hierarchy for layer classes.
 *
 *  To make it easy to subclass from classes in the HyPerLayer hierarchy,
 *  please follow the guidelines below when adding subclasses to the HyPerLayer hierarchy:
 *
 *  For a class named DerivedLayer that is derived from a class named BaseLayer,
 *  the .hpp file should have
namespace PV {
class DerivedLayer : public BaseLayer {
public:
  DerivedLayer(arguments); // The constructor called by
  // other methods
protected:
  DerivedLayer();
  int initialize(arguments);
  // other methods and member variables
private:
  int initialize_base();
  // other methods and member variables
};
}
 *
 * The .cpp file should have
namespace PV {
DerivedLayer::DerivedLayer() {
  initialize_base();
  // initialize(arguments) should *not* be called by the protected constructor.
}
DerivedLayer::DerivedLayer(arguments, generally includes the layer's name and the parent HyPerCol) {
  initialize_base();
  initialize(arguments);
}
DerivedLayer::initialize_base() {
  // the most basic initializations.  Don't call any virtual methods,
  // or methods that call virtual methods, etc. from initialize_base();
}
DerivedLayer::initialize(arguments) {
  // DerivedLayer-specific initializations that need to precede BaseClass initialization, if any
  BaseClass::initialize(BaseClass initialization arguments);
  // DerivedLayer-specific initializations
}

  // other DerivedLayer methods
}
 */

#ifndef HYPERLAYER_HPP_
#define HYPERLAYER_HPP_

#include <layers/accumulate_functions.h>
#include <layers/PVLayerCube.h>
#include <layers/BaseLayer.hpp>
#include <columns/DataStore.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/InterColComm.hpp>
#include <columns/Random.hpp>
#include <io/LayerProbe.hpp>
#include <io/fileio.hpp>
#include <include/pv_common.h>
#include <include/pv_types.h>
#include <utils/Timer.hpp>


#ifdef PV_USE_CUDA
#  undef PV_USE_CUDA
#  include <layers/updateStateFunctions.h>
#  define PV_USE_CUDA
#elif defined(PV_USE_OPENCL)
#  undef PV_USE_OPENCL
#  include <layers/updateStateFunctions.h>
#  define PV_USE_OPENCL
#else
#  include <layers/updateStateFunctions.h>
#endif //PV_USE_OPENCL

#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif //PV_USE_OPENMP_THREADS


#ifdef PV_USE_OPENCL
#define PV_CL_COPY_BUFFERS 0
#define PV_CL_EVENTS 1
#include <arch/opencl/CLKernel.hpp>
#define EV_GSYN 0
#define EV_ACTIVITY 1
#define EV_HPL_PHI_E 0
#define EV_HPL_PHI_I 1
#endif //PV_USE_OPENCL

#ifdef PV_USE_CUDA
#include <arch/cuda/CudaKernel.hpp>
#include <arch/cuda/CudaBuffer.hpp>
#include <arch/cuda/CudaTimer.hpp>
#endif //PV_USE_CUDA

#ifdef PV_USE_OPENCL
#include <arch/opencl/CLTimer.hpp>
#endif //PV_USE_OPENCL

#include <vector>



// default constants
#define HYPERLAYER_FEEDBACK_DELAY 1
#define HYPERLAYER_FEEDFORWARD_DELAY 0


namespace PV {

class InitV;
class PVParams;
class BaseConnection;

typedef enum TriggerBehaviorTypeEnum { NO_TRIGGER, UPDATEONLY_TRIGGER, RESETSTATE_TRIGGER } TriggerBehaviorType;

class HyPerLayer : public BaseLayer{


protected:

   /** 
    * List of parameters needed from the HyPerLayer class
    * @name HyPerLayer Parameters
    * @{
    */

   virtual void ioParam_dataType(enum ParamsIOFlag ioFlag);
   
   /**
    * @brief updateGpu: When compiled using CUDA or OpenCL GPU acceleration, this flag tells whether this layer's updateState method should use the GPU.
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
    * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);

   /**
    * @brief initVType: Specifies how to initialize the V buffer. 
    * @details Possible choices include
    * - @link InitV::ioParamGroup_ConstantV ConstantV@endlink: Sets V to a constant value
    * - @link InitV::ioParamGroup_ZeroV ZeroV@endlink: Sets V to zero
    * - @link InitV::ioParamGroup_UniformRandomV UniformRandomV@endlink: Sets V with a uniform distribution
    * - @link InitV::ioParamGroup_GaussianRandomV GaussianRandomV@endlink: Sets V with a gaussian distribution
    * - @link InitV::ioparamGroup_InitVFromFile InitVFromFile@endlink: Sets V to specified pvp file
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
    * If set to NULL or the empty string, the layer does not trigger but updates its state on every timestep.
    */
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);

   // TODO: triggerOffset is measured in units of simulation time, not timesteps.  How does adaptTimeStep affect
   // the triggering time?
   /**
    * @brief triggerOffset: If triggerLayer is set, triggers <triggerOffset> timesteps before target trigger
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
    * @brief triggerResetLayerName: If triggerLayerName is set, this parameter specifies the layer to use for updating
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
    * @brief sparseLayer: Specifies if the layer should be considered sparese for optimization and output
    */
   virtual void ioParam_sparseLayer(enum ParamsIOFlag ioFlag);

   /**
    * @brief writeSparseValues: If sparseLayer is set, specifies if the pvp file should write sparse value file
    */
   virtual void ioParam_writeSparseValues(enum ParamsIOFlag ioFlag);
   /** @} */


private:
   int initialize_base();

protected:

   // only subclasses can be constructed directly
   HyPerLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int initClayer();

   virtual int allocateClayerBuffers();
   int setLayerLoc(PVLayerLoc * layerLoc, float nxScale, float nyScale, int nf, int numBatches);
   virtual int allocateBuffers();
   virtual int allocateGSyn();

   /*
    * Allocates a buffer of the given length.  The membrane potential and activity buffer, among others, are created using allocateBuffer.
    * To free a buffer created with this method, call freeBuffer().
    */
   template <typename T>
   int allocateBuffer(T ** buf, int bufsize, const char * bufname);

   /**
    * Allocates a restricted buffer (that is, buffer's length is getNumNeuronsAllBatches()).
    */
   int allocateRestrictedBuffer(pvdata_t ** buf, const char * bufname);

   /**
    * Allocates an extended buffer (that is, buffer's length is getNumExtendedAllBatches()).
    */
   int allocateExtendedBuffer(pvdata_t ** buf, const char * bufname);

   int allocateCube();
   virtual int allocateV();
   virtual int allocateActivity();
   virtual int allocatePrevActivity();
   virtual int setInitialValues();
   virtual int initializeV();
   virtual int initializeActivity();
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readActivityFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readVFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readDelaysFromCheckpoint(const char * cpDir, double * timeptr);
#ifdef PV_USE_CUDA
   virtual int copyInitialStateToGPU();
#endif // PV_USE_CUDA
   char * pathInCheckpoint(const char * cpDir, const char * suffix);
   int readDataStoreFromFile(const char * filename, InterColComm * comm, double * timed);
   int incrementNBands(int * numCalls);
   int writeDataStoreToFile(const char * filename, InterColComm * comm, double dtime);
   //virtual int calcActiveIndices();
   void calcNumExtended();
   
   /**
    * Called by updateStateWrapper when updating the state in the usual way
    * (as opposed to being triggered when triggerBehavior is resetStateOnTrigger).
    * It calls either updateState or updateStateGPU.  It also starts and stops the update timer.
    */
   virtual int callUpdateState(double timed, double dt);
   
   /**
    * Returns true if the trigger behavior is resetStateOnTrigger and the layer was triggered.
    */
   virtual bool needReset(double timed, double dt);
   
   /**
    * Called instead of updateState when triggerBehavior is "resetStateOnTrigger" and a triggering event occurs.
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
   int freeBuffer(T ** buf);

   /**
    * Frees a buffer created by allocateRestrictedBuffer().
    * Note that there is no checking whether the buffer was created by allocateRestrictedBuffer(),
    * or any other allocateBuffer()-related method.
    */
   int freeRestrictedBuffer(pvdata_t ** buf);

   /**
    * Frees a buffer created by allocateRestrictedBuffer().
    * Note that there is no checking whether the buffer was created by allocateExtendedBuffer(),
    * or any other allocateBuffer()-related method.
    */
   int freeExtendedBuffer(pvdata_t ** buf);

public:
   pvdata_t * getActivity()          {return clayer->activity->data;} // TODO: access to clayer->activity->data should not be public
   virtual double calcTimeScale(int batchIdx)          {return -1.0;};
   virtual double getTimeScale(int batchIdx)      {return -1.0;};
   virtual bool activityIsSpiking() = 0; // Pure virtual method so that subclasses are forced to implement it.
   PVDataType getDataType()          {return dataType;}
protected:

   /**
    * The function that calls all ioParam functions
    */
   virtual int  ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   static int equalizeMargins(HyPerLayer * layer1, HyPerLayer * layer2);

   int freeClayer();

public:

   virtual ~HyPerLayer() = 0;
   int initializeState(); // Not virtual since all layers should respond to initializeFromCheckpointFlag and (deprecated) restartFlag in the same way.
                          // initializeState calls the virtual methods readStateFromCheckpoint(), and setInitialValues().

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

   void synchronizeMarginWidth(HyPerLayer * layer);

   // TODO The three routines below shouldn't be public, but HyPerCol needs to call them, so for now they are.
   void setInitInfoCommunicatedFlag() {initInfoCommunicatedFlag = true;}
   void setDataStructuresAllocatedFlag() {dataStructuresAllocatedFlag = true;}
   void setInitialValuesSetFlag() {initialValuesSetFlag = true;}

   bool getInitInfoCommunicatedFlag() {return initInfoCommunicatedFlag;}
   bool getDataStructuresAllocatedFlag() {return dataStructuresAllocatedFlag;}
   bool getInitialValuesSetFlag() {return initialValuesSetFlag;}

   int ioParams(enum ParamsIOFlag ioFlag);

   // TODO - make protected
   PVLayer  * clayer;

   // ************************************************************************************//
   // interface for public methods for controlling HyPerLayer cellular and synaptic dynamics
   // (i.e. methods for receiving synaptic input, updating internal state, publishing output)
   // ************************************************************************************//
   virtual int recvAllSynapticInput(); // Calls recvSynapticInput for each conn and each arborID

   //An updateState wrapper that determines if updateState needs to be called
   virtual int updateStateWrapper (double time, double dt);
   virtual int updateState (double time, double dt);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int updateStateGpu (double time, double dt);
#endif
   /**
    * A virtual function to determine if callUpdateState method needs to be called
    * Default behavior is dependent on the triggering method.
    * If there is no triggering, always returns true.
    * If there is triggering and the trigger behavior is updateOnlyOnTrigger, returns true only when there is a triggering event.
    * If there is triggering and the trigger behavior is resetStateOnTrigger, returns true only when there is not a trigger event.
    * @param time The current timestep of the run
    * @param dt The current non-adaptive dt of the run
    * @return Returns if the update needs to happen
    */
   virtual bool needUpdate(double time, double dt);
   /**
    * A function to update nextUpdateTime of the layer based on trigger
    */
   virtual int updateNextUpdateTime();
   /**
    * A function to set nextUpdateTime to a specific time
    */
   virtual void setNextUpdateTime(double in){nextUpdateTime = in;}
   /**
    * A function to return the interval between times when updateState is needed.
    */
   virtual double getDeltaUpdateTime();
   /**
    * A function to return the interval between triggering times.  A negative value means that the layer never triggers
    * (either there is no triggerLayer or the triggerLayer never updates).
    */
   virtual double getDeltaTriggerTime();
   /**
    * A function to update the time that the next trigger is expected to occur.
    */
   virtual int updateNextTriggerTime();
   virtual int publish(InterColComm * comm, double time);
   virtual int resetGSynBuffers(double timef, double dt);
   // ************************************************************************************//

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to next time step
   virtual int waitOnPublish(InterColComm * comm);

   virtual int updateBorder(double time, double dt);

   virtual int updateAllActiveIndices();
   virtual int updateActiveIndices();
   int resetBuffer(pvdata_t * buf, int numItems);

   static bool localDimensionsEqual(PVLayerLoc const * loc1, PVLayerLoc const * loc2);
   int mirrorInteriorToBorder(int whichBorder, PVLayerCube * cube, PVLayerCube * borderCube);
   int mirrorInteriorToBorder(PVLayerCube * cube, PVLayerCube * borderCube);

   virtual int checkpointRead(const char * cpDir, double * timeptr); // (const char * cpDir, double * timed);
   virtual int checkpointWrite(const char * cpDir);
   virtual int writeTimers(FILE* stream);
   // TODO: readBufferFile and writeBufferFile have to take different types of buffers.  Can they be templated?
   template <typename T>
   static int readBufferFile(const char * filename, InterColComm * comm, double * timed, T ** buffers, int numbands, bool extended, const PVLayerLoc * loc);
   template <typename T>
   static int writeBufferFile(const char * filename, InterColComm * comm, double dtime, T ** buffers, int numbands, bool extended, const PVLayerLoc * loc);

   virtual int outputState(double timef, bool last=false);
   virtual int writeActivity(double timed);
   virtual int writeActivitySparse(double timed, bool includeValues);

   virtual int insertProbe(LayerProbe * probe);
   int outputProbeParams();

   int getNumProbes() { return numProbes; }
   LayerProbe * getProbe(int n) { return (n>=0 && n<numProbes) ? probes[n] : NULL; }

   // TODO: should the mirroring functions be static?  Why are they virtual?
   virtual int mirrorToNorthWest(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToNorth    (PVLayerCube * dest, PVLayerCube* src);
   virtual int mirrorToNorthEast(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToWest     (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToEast     (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouthWest(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouth    (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouthEast(PVLayerCube * dest, PVLayerCube * src);

   // Public access functions:

   int getNumNeurons()               {return clayer->numNeurons;}
   int getNumExtended()              {return clayer->numExtended;}
   int getNumNeuronsAllBatches()     {return clayer->numNeuronsAllBatches;}
   int getNumExtendedAllBatches()    {return clayer->numExtendedAllBatches;}


   int getNumGlobalNeurons()         {const PVLayerLoc * loc = getLayerLoc(); return loc->nxGlobal*loc->nyGlobal*loc->nf;}
   int getNumGlobalExtended()        {const PVLayerLoc * loc = getLayerLoc(); return (loc->nxGlobal+loc->halo.lt+loc->halo.rt)*(loc->nyGlobal+loc->halo.dn+loc->halo.up)*loc->nf;}
   int getNumDelayLevels()           {return numDelayLevels;}

   int  getLayerId()                 {return layerId;}
   void setLayerId(int id)           {layerId = id;}
   int increaseDelayLevels(int neededDelay);
   virtual int requireMarginWidth(int marginWidthNeeded, int * marginWidthResult, char axis);
   virtual int requireChannel(int channelNeeded, int * numChannelsResult);

   PVLayer*  getCLayer()             {return clayer;}
   pvdata_t * getV()                 {return clayer->V;}           // name query
   int getNumChannels()              {return numChannels;}
   pvdata_t * getChannel(ChannelType ch) {                         // name query
      return (ch < this->numChannels && ch >= 0) ? GSyn[ch] : NULL;
   }
   virtual float getChannelTimeConst(enum ChannelType channel_type){return 0.0f;};
   int getXScale()                   {return clayer->xScale;}
   int getYScale()                   {return clayer->yScale;}

   //int getNumActive()                {return clayer->numActive;}

   bool useMirrorBCs()               {return this->mirrorBCflag;}
   pvdata_t getValueBC() {return this->valueBC;}

   bool getSparseFlag()             {return this->sparseLayer;}

   int getPhase()                    {return this->phase;}

   char const * getOutputStatePath();
   int flushOutputStateStream();

   // implementation of LayerDataInterface interface
   //
   const pvdata_t   * getLayerData(int delay=0);
   const PVLayerLoc * getLayerLoc()  { return &(clayer->loc); }
   bool isExtended()                 { return true; }

   double getLastUpdateTime() { return lastUpdateTime; }
   double getNextUpdateTime() { return nextUpdateTime; }

   float getMaxRate() {return maxRate;}

protected:

   int openOutputStateFile();
   /* static methods called by updateState({long_argument_list})*/

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int runUpdateKernel();
   virtual int doUpdateStateGpu(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * GSynHead);
#endif
   virtual int doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * GSynHead);
   virtual int setActivity();
   void freeChannels();

   int layerId;                 // unique ID that identifies layer in its parent HyPerCol

   int numChannels;             // number of channels
   pvdata_t ** GSyn;            // of dynamic length numChannels

   float nxScale, nyScale;        // Size of layer relative to column
   int numFeatures;
   int xmargin, ymargin;

   bool initializeFromCheckpointFlag; // Whether to load initial state using directory parent->getInitializeFromCheckpoint()
   bool restartFlag;

   int numProbes;
   LayerProbe ** probes;

   int phase;                   // All layers with phase 0 get updated before any with phase 1, etc.
   int numDelayLevels;          // The number of timesteps in the datastore ring buffer to store older timesteps for connections with delays

   bool mirrorBCflag;           // true when mirror BC are to be applied
   pvdata_t valueBC; // If mirrorBCflag is false, the value of A to fill extended cells with

   int ioAppend;                // controls opening of binary files
   double initialWriteTime;             // time of next output
   double writeTime;             // time of next output
   double writeStep;             // output time interval
   PV_Stream * outputStateStream;       // activity generated by outputState

   bool sparseLayer; // if true, only nonzero activities are saved; if false, all values are saved.
   bool writeSparseValues; // if true, sparseLayer writes index-value pairs.  if false, sparseLayer writes indices only and values are assumed to be 1.  Not used if sparseLayer is false
   int writeActivityCalls;      // Number of calls to writeActivity (written to nbands in the header of the a%d.pvp file)
   int writeActivitySparseCalls; // Number of calls to writeActivitySparse (written to nbands in the header of the a%d.pvp file)

   int * marginIndices;   // indices of neurons in margin
   int numMargin;         // number of neurons in margin
   float maxRate;         // Maximum rate of activity.  HyPerLayer sets to 1/dt during initialize(); derived classes should override in their own initialize method after calling HyPerLayer's, if needed.

   unsigned int rngSeedBase; // The starting seed for rng.  The parent HyPerCol reserves {rngSeedbase, rngSeedbase+1,...rngSeedbase+neededRNGSeeds-1} for use by this layer

   bool initInfoCommunicatedFlag;
   bool dataStructuresAllocatedFlag;
   bool initialValuesSetFlag;

   InitV * initVObject;

   HyPerLayer ** synchronizedMarginWidthLayers;
   int numSynchronizedMarginWidthLayers;

   //Trigger-related parameters
   //  Although triggerFlag was deprecated as a params file parameter, it remains as a member variable to allow quick testing of whether we're triggering.  It is set during ioParam_triggerLayerName.
   bool triggerFlag; // Whether the layer has different behavior in response to another layer's update.
   char* triggerLayerName; // The layer that triggers different behavior.  To turn triggering off, set this parameter to NULL or ""
   char * triggerBehavior; // Specifies how to respond to a trigger.  Current values are "updateOnlyOnTrigger" or "resetStateOnTrigger"
   TriggerBehaviorType triggerBehaviorType;
   char * triggerResetLayerName; // If triggerBehavior is "resetStateOnTrigger", specifies the layer to use in resetting values.
   double triggerOffset; // Adjust the timestep when the trigger is receieved by this amount; must be >=0.  A positive value means the trigger occurs before the triggerLayerName layer updates.
   HyPerLayer* triggerLayer;
   HyPerLayer * triggerResetLayer;

   char* dataTypeString;
   PVDataType dataType;


   double lastUpdateTime; // The most recent time that the layer's activity is updated, used as a cue for publisher to exchange borders
   double nextUpdateTime; // The timestep to update next
   double nextTriggerTime; // The timestep when triggerLayer is next expected to trigger.

   pvdata_t ** thread_gSyn; //Accumulate buffer for each thread, only used if numThreads > 1
   std::vector<BaseConnection *> recvConns;

   // OpenCL variables
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
public:

   virtual void syncGpu();
   virtual float addGpuTimers();

   void copyAllGSynToDevice();
   void copyAllGSynFromDevice();
   void copyAllVFromDevice();
   void copyAllActivityFromDevice();

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceV(){
#endif

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceV(){
#endif
      return d_V;
   }

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceGSyn() {
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceGSyn() {
#endif
      return d_GSyn;
   }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   PVCuda::CudaBuffer * getCudnnGSyn(){
      return cudnn_GSyn;
   }
#endif

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceActivity(){
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceActivity(){
#endif
      return d_Activity;
   }

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceDatastore(){
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceDatastore(){
#endif
      return d_Datastore;
   }

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceActiveIndices(){
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceActiveIndices(){
#endif
      return d_ActiveIndices;
   }

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceNumActive(){
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceNumActive(){
#endif
      return d_numActive;
   }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   PVCuda::CudaBuffer * getCudnnDatastore(){
      return cudnn_Datastore;
   }
#endif

   void setAllocDeviceV(){
      allocDeviceV = true;
   }
   void setAllocDeviceGSyn(){
      allocDeviceGSyn = true;
   }

   void setAllocDeviceActivity(){
      allocDeviceActivity = true;
   }

   void setAllocDeviceDatastore(){
      allocDeviceDatastore= true;
   }

   void setAllocDeviceActiveIndices(){
      allocDeviceActiveIndices = true;
   }

   bool getUpdatedDeviceActivityFlag(){
      return updatedDeviceActivity;
   }

   void setUpdatedDeviceActivityFlag(bool in){
      updatedDeviceActivity = in;
   }

   bool getUpdatedDeviceDatastoreFlag(){
      return updatedDeviceDatastore;
   }

   void setUpdatedDeviceDatastoreFlag(bool in){
      updatedDeviceDatastore = in;
   }

   bool getUpdatedDeviceGSynFlag(){
      return updatedDeviceGSyn;
   }

   void setUpdatedDeviceGSynFlag(bool in){
      updatedDeviceGSyn = in;
   }

   bool getRecvGpu(){
      return recvGpu;
   }

   bool getUpdateGpu(){
      return updateGpu;
   }

#ifdef PV_USE_OPENCL
   void clFinishGSyn(){
      if(allocDeviceGSyn && d_GSyn){
         d_GSyn->finish(); //This should take care of every command in the queue
      }
   }
   void clFinishActivity(){
      if(allocDeviceActivity){
         d_Activity->finish();
      }
   }
   cl_event * getRecvSynStartEvent() { return gpu_recvsyn_timer->getStartEvent(); }
#endif // PV_USE_OPENCL

protected:

   virtual int allocateUpdateKernel();
   virtual int allocateDeviceBuffers();
   // OpenCL buffers and their corresponding flags
   //
   
   bool allocDeviceV;
   bool allocDeviceGSyn;         // array of channels to allocate
   bool allocDeviceActivity;
   bool allocDeviceDatastore;
   bool allocDeviceActiveIndices;
   bool updatedDeviceActivity;
   bool updatedDeviceDatastore;
   bool updatedDeviceGSyn;
   bool recvGpu;
   bool updateGpu;

#ifdef PV_USE_OPENCL
   CLBuffer * d_V;
   CLBuffer * d_GSyn;         
   CLBuffer * d_Activity;
   CLBuffer * d_Datastore;
   CLBuffer * d_numActive;
   CLBuffer * d_ActiveIndices;
   CLKernel * krUpdate;
#endif

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * d_V;
   PVCuda::CudaBuffer * d_GSyn;      
   PVCuda::CudaBuffer * d_Activity;
   PVCuda::CudaBuffer * d_Datastore;
   PVCuda::CudaBuffer * d_numActive;
   PVCuda::CudaBuffer * d_ActiveIndices;
   PVCuda::CudaKernel * krUpdate;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer * cudnn_GSyn; 
   PVCuda::CudaBuffer * cudnn_Datastore;
#endif //PV_USE_CUDNN
#endif //PV_USE_CUDA

#endif //PV_USE_CUDA || PV_USE_OPENCL

protected:
   Timer * update_timer;
   Timer * recvsyn_timer;
   Timer * recvsyn_calc_timer;
   Timer * publish_timer;
   Timer * timescale_timer;
   Timer * io_timer;

#ifdef PV_USE_CUDA
   PVCuda::CudaTimer * gpu_recvsyn_timer;
   PVCuda::CudaTimer * gpu_update_timer;
#endif

#ifdef PV_USE_OPENCL
   CLTimer * gpu_recvsyn_timer;
   CLTimer * gpu_update_timer;
#endif

//Removed fields

//#ifdef PV_USE_CUDNN
   //PVCuda::CudaTimer * permute_weights_timer;
   //PVCuda::CudaTimer * permute_preData_timer;
   //PVCuda::CudaTimer * permute_postGSyn_timer;
//#endif
   //virtual int allocateActiveIndices();
   //static int copyToBuffer(pvdata_t * buf, const pvdata_t * data,
   //                        const PVLayerLoc * loc, bool extended, float scale);
   //static int copyToBuffer(unsigned char * buf, const pvdata_t * data,
   //                        const PVLayerLoc * loc, bool extended, float scale);

   //template <typename T>
   //static int copyFromBuffer(const T * buf, T * data,
   //                          const PVLayerLoc * loc, bool extended, T scale);

   //static int copyFromBuffer(const unsigned char * buf, pvdata_t * data,
   //                          const PVLayerLoc * loc, bool extended, float scale);
   ///** returns the number of neurons in layer (for borderId=0) or a border region **/
   //virtual int numberOfNeurons(int borderId);
   //virtual int gatherToInteriorBuffer(unsigned char * buf);

   //Labels deprecated 6/16/15
   //virtual int label(int k);
   //virtual int * getMarginIndices();
   //virtual int getNumMargin();

//   int getFeedbackDelay(){return feedbackDelay;};
//   int getFeedforwardDelay(){return feedforwardDelay;};
   //Labels deprecated 6/16/15
   //int * labels;                // label for the feature a neuron is tuned to
   //virtual int allocateReceivePostKernel();

//   CLKernel * krUpdate;        // CL kernel for update state call
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
