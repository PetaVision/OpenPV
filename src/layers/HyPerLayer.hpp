/*
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

#include "../layers/accumulate_functions.h"
#include "../layers/PVLayerCube.h"
#include "../layers/LayerDataInterface.hpp"
#include "../columns/DataStore.hpp"
#include "../columns/HyPerCol.hpp"
#include "../columns/InterColComm.hpp"
#include "../io/LayerProbe.hpp"
#include "../io/fileio.hpp"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../utils/Timer.hpp"

#ifndef PV_USE_OPENCL
#  include "../layers/updateStateFunctions.h"
#else
#  undef PV_USE_OPENCL
#  include "../layers/updateStateFunctions.h"
#  define PV_USE_OPENCL
#endif //PV_USE_OPENCL

#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif //PV_USE_OPENMP_THREADS


#ifdef PV_USE_OPENCL
#define PV_CL_COPY_BUFFERS 0
#define PV_CL_EVENTS 1
#include "../arch/opencl/CLKernel.hpp"
#define EV_GSYN 0
#define EV_ACTIVITY 1
#define EV_HPL_PHI_E 0
#define EV_HPL_PHI_I 1
#endif //PV_USE_OPENCL

#ifdef PV_USE_CUDA
#include "../arch/cuda/CudaKernel.hpp"
#include "../arch/cuda/CudaBuffer.hpp"
#include "../arch/cuda/CudaTimer.hpp"
#endif //PV_USE_CUDA

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLTimer.hpp"
#endif //PV_USE_OPENCL

#include <vector>



// default constants
#define HYPERLAYER_FEEDBACK_DELAY 1
#define HYPERLAYER_FEEDFORWARD_DELAY 0


namespace PV {

class InitV;
class PVParams;

class HyPerLayer : public LayerDataInterface {

private:
   int initialize_base();

protected:

   // only subclasses can be constructed directly
   HyPerLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int initClayer();

   virtual int allocateClayerBuffers();
   int setLayerLoc(PVLayerLoc * layerLoc, float nxScale, float nyScale, int nf);
   virtual int allocateBuffers();
   virtual int allocateGSyn();

   template <typename T>
   int allocateBuffer(T ** buf, int bufsize, const char * bufname);

   int allocateCube();
   virtual int allocateV();
   virtual int allocateActivity();
   virtual int allocateActiveIndices();
   virtual int allocatePrevActivity();
   virtual int setInitialValues();
   virtual int initializeV();
   virtual int initializeActivity();
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readActivityFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readVFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readDelaysFromCheckpoint(const char * cpDir, double * timeptr);
   char * pathInCheckpoint(const char * cpDir, const char * suffix);
   int readDataStoreFromFile(const char * filename, InterColComm * comm, double * timed);
   int incrementNBands(int * numCalls);
   int writeDataStoreToFile(const char * filename, InterColComm * comm, double dtime);
   virtual int calcActiveIndices();
   void calcNumExtended();
public:
   pvdata_t * getActivity()          {return clayer->activity->data;}
   virtual double calcTimeScale()          {return -1.0;};
   virtual double getTimeScale()      {return -1.0;};
protected:

   virtual int  ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nxScale(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nyScale(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nf(enum ParamsIOFlag ioFlag);
   virtual void ioParam_marginWidth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_phase(enum ParamsIOFlag ioFlag);
   virtual void ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_valueBC(enum ParamsIOFlag ioFlag);
   virtual void ioParam_restart(enum ParamsIOFlag ioFlag); // Deprecating in favor of initializeFromCheckpointFlag?
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag);
   virtual void ioParam_triggerOffset(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeSparseActivity(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeSparseValues(enum ParamsIOFlag ioFlag);

   static int equalizeMargins(HyPerLayer * layer1, HyPerLayer * layer2);

   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int arborID);
   virtual int recvSynapticInputFromPost(HyPerConn * conn, const PVLayerCube * activity, int arborID);
   void recvOnePreNeuronActivity(HyPerConn * conn, int patchIndex, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr);

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int recvSynapticInputGpu(HyPerConn * conn, const PVLayerCube * cube, int arborID, bool firstRun);
   virtual int recvSynapticInputFromPostGpu(HyPerConn * conn, const PVLayerCube * activity, int arborID, bool firstRun);
#endif 


   int freeClayer();

public:

   virtual ~HyPerLayer() = 0;
   int initializeState(); // Not virtual since all layers should respond to initializeFromCheckpointFlag and (deprecated) restartFlag in the same way.
                          // initializeState calls the virtual methods readStateFromCheckpoint(), readState() (deprecated), and setInitialValues().

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

   static int copyToBuffer(pvdata_t * buf, const pvdata_t * data,
                           const PVLayerLoc * loc, bool extended, float scale);
   static int copyToBuffer(unsigned char * buf, const pvdata_t * data,
                           const PVLayerLoc * loc, bool extended, float scale);

   template <typename T>
   static int copyFromBuffer(const T * buf, T * data,
                             const PVLayerLoc * loc, bool extended, T scale);

   static int copyFromBuffer(const unsigned char * buf, pvdata_t * data,
                             const PVLayerLoc * loc, bool extended, float scale);

   // TODO - make protected
   PVLayer  * clayer;

   // ************************************************************************************//
   // interface for public methods for controlling HyPerLayer cellular and synaptic dynamics
   // (i.e. methods for receiving synaptic input, updating internal state, publishing output)
   // ************************************************************************************//
   virtual int recvAllSynapticInput(); // Calls recvSynapticInput for each conn and each arborID
   //Method to see if the neuron is in the window. Default window id is mapped to the arbor id. Parent class is always true, and can be overwritten 
   virtual bool inWindowExt(int windowId, int neuronIdxExt) {return true;};
   virtual bool inWindowRes(int windowId, int neuronIdxRes) {return true;}; 
   //Returns number of windows, with a default of 1 window for the entire layer
   virtual int getNumWindows(){return 1;}

   //An updateState wrapper that determines if updateState needs to be called
   virtual int updateStateWrapper (double time, double dt);
   virtual int updateState (double time, double dt);
   /**
    * A virtual function to determine if updateState method needs to be called
    * Default behaviour is dependent on the flag triggerFlag. If true, will call attached trigger layer's needUpdate
    * If triggerFlag is false, this function will return true
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
    * A function to compute the change in update time
    */
   virtual double getDeltaUpdateTime();
   virtual int publish(InterColComm * comm, double time);
   virtual int resetGSynBuffers(double timef, double dt);
   // ************************************************************************************//

   // mpi public wait method to ensure all targets have received synaptic input before proceeding to next time step
   virtual int waitOnPublish(InterColComm * comm);

   virtual int updateBorder(double time, double dt);

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

   virtual int readState (double * timeptr);
   virtual int outputState(double timef, bool last=false);
   virtual int writeActivity(double timed);
   virtual int writeActivitySparse(double timed, bool includeValues);

   virtual int insertProbe(LayerProbe * probe);
   int outputProbeParams();

   int getNumProbes() { return numProbes; }
   LayerProbe * getProbe(int n) { return (n>=0 && n<numProbes) ? probes[n] : NULL; }

   /** returns the number of neurons in layer (for borderId=0) or a border region **/
   virtual int numberOfNeurons(int borderId);

   // TODO: should the mirroring functions be static?  Why are they virtual?
   virtual int mirrorToNorthWest(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToNorth    (PVLayerCube * dest, PVLayerCube* src);
   virtual int mirrorToNorthEast(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToWest     (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToEast     (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouthWest(PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouth    (PVLayerCube * dest, PVLayerCube * src);
   virtual int mirrorToSouthEast(PVLayerCube * dest, PVLayerCube * src);

   const char * getOutputFilename(char * buf, const char * dataName, const char * term);

   // Public access functions:

   const char * getName()            {return name;}

   int getNumNeurons()               {return clayer->numNeurons;}
   int getNumExtended()              {return clayer->numExtended;}
   int getNumGlobalNeurons()         {const PVLayerLoc * loc = getLayerLoc(); return loc->nxGlobal*loc->nyGlobal*loc->nf;}
   int getNumGlobalExtended()        {const PVLayerLoc * loc = getLayerLoc(); return (loc->nxGlobal+loc->halo.lt+loc->halo.rt)*(loc->nyGlobal+loc->halo.dn+loc->halo.up)*loc->nf;}
   int getNumDelayLevels()           {return numDelayLevels;}

   int  getLayerId()                 {return layerId;}
   PVLayerType getLayerType()        {return clayer->layerType;}
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

   HyPerCol* getParent()             {return parent;}
   void setParent(HyPerCol* parent)  {this->parent = parent;}

   bool useMirrorBCs()               {return this->mirrorBCflag;}
   pvdata_t getValueBC() {return this->valueBC;}

   bool getSpikingFlag()             {return this->writeSparseActivity;}

   int getPhase()                    {return this->phase;}

   // implementation of LayerDataInterface interface
   //
   const pvdata_t   * getLayerData(int delay=0);
   const PVLayerLoc * getLayerLoc()  { return &(clayer->loc); }
   bool isExtended()                 { return true; }

   //TODO don't make this virtual
   virtual double getLastUpdateTime();
   double getNextUpdateTime() { return nextUpdateTime;}

   virtual int gatherToInteriorBuffer(unsigned char * buf);

   virtual int label(int k);

   virtual int * getMarginIndices();
   virtual int getNumMargin();
   float getConvertToRateDeltaTimeFactor(HyPerConn* conn);
   float getMaxRate() {return maxRate;}

//   int getFeedbackDelay(){return feedbackDelay;};
//   int getFeedforwardDelay(){return feedforwardDelay;};

protected:

   int openOutputStateFile();
   /* static methods called by updateState({long_argument_list})*/
   virtual int doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * GSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active);
   virtual int setActivity();
   void freeChannels();

   HyPerCol * parent;

   char * name;                 // well known name of layer

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

   int * labels;                // label for the feature a neuron is tuned to

   bool mirrorBCflag;           // true when mirror BC are to be applied
   pvdata_t valueBC; // If mirrorBCflag is false, the value of A to fill extended cells with

   int ioAppend;                // controls opening of binary files
   double initialWriteTime;             // time of next output
   double writeTime;             // time of next output
   double writeStep;             // output time interval

   bool writeSparseActivity; // if true, only nonzero activities are saved; if false, all values are saved.
   bool writeSparseValues; // if true, writeSparseActivity writes index-value pairs.  if false, writeSparseActivity writes indices only and values are assumed to be 1.  Not used if writeSparseActivity is false
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

   //A flag that determines if the layer is a trigger layer and needs to follow another layer's lastUpdateTime.
   bool triggerFlag;
   char* triggerLayerName;
   double triggerOffset;
   HyPerLayer* triggerLayer;


   double lastUpdateTime; // The most recent time that the layer's activity is updated, used as a cue for publisher to exchange borders
   double nextUpdateTime; // The timestep to update next

//   int feedforwardDelay;  // minimum delay required for a change in the input to potentially influence this layer
//   int feedbackDelay;     // minimum delay required for a change in this layer to potentially influence itself via feedback loop
private:

   pvdata_t ** thread_gSyn; //Accumulate buffer for each thread, only used if numThreads > 1
   std::vector<HyPerConn*> recvConns;

   // OpenCL variables
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
public:

   virtual float syncGpu();

   void copyAllGSynToDevice();
   void copyAllGSynFromDevice();

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceV(){
#endif

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceV(){
#endif
      return d_V;
   }

#ifdef PV_USE_OPENCL
   CLBuffer * getDeviceGSyn(ChannelType ch) {
#endif

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * getDeviceGSyn(ChannelType ch) {
#endif
      return (ch < this->numChannels && ch >= 0) ? d_GSyn[ch] : NULL;
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

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   PVCuda::CudaBuffer * getCudnnActivity(){
      return cudnn_Activity;
   }
#endif

   void setAllocDeviceV(){
      allocDeviceV = true;
   }
   void setAllocDeviceGSyn(ChannelType ch){
      if(ch >= 0 && ch < this->numChannels){
         allocDeviceGSyn[ch] = true;
      }
   }
   void setAllocDeviceActivity(){
      allocDeviceActivity = true;
   }

   bool getUpdatedDeviceActivityFlag(){
      return updatedDeviceActivity;
   }

   void setUpdatedDeviceActivityFlag(bool in){
      updatedDeviceActivity = in;
   }

   bool getRecvGpu(){
      return recvGpu;
   }

#ifdef PV_USE_OPENCL
   void clFinishGSyn(){
      for(int i = 0; i < this->numChannels; i++){
         if(allocDeviceGSyn[i] && d_GSyn[i]){
            d_GSyn[i]->finish(); //This should take care of every command in the queue
            break;
         }
      }
   }
   void clFinishActivity(){
      if(allocDeviceActivity && allocDeviceActivity){
         d_Activity->finish();
      }
   }
#endif
   //void setAllocKrRecvPost(){
   //   allocKrRecvPost = true;
   //}
//   void startTimer() {recvsyn_timer->start();}
//   void stopTimer() {recvsyn_timer->stop();}
//
protected:

   virtual int allocateDeviceBuffers();
   //virtual int allocateReceivePostKernel();

//   CLKernel * krUpdate;        // CL kernel for update state call

   // OpenCL buffers and their corresponding flags
   //
   
   bool allocDeviceV;
   bool* allocDeviceGSyn;         // array of channels to allocate
   bool allocDeviceActivity;
   bool updatedDeviceActivity;
   bool recvGpu;

#ifdef PV_USE_OPENCL
   CLBuffer * d_V;
   CLBuffer **d_GSyn;         // of dynamic length numChannels
   CLBuffer * d_Activity;
#endif

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer * d_V;
   PVCuda::CudaBuffer **d_GSyn;         // of dynamic length numChannels
   PVCuda::CudaBuffer * d_Activity;
#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer * cudnn_GSyn;         // of dynamic length numChannels
   PVCuda::CudaBuffer * cudnn_Activity;
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
#ifdef PV_USE_CUDNN
   //PVCuda::CudaTimer * permute_weights_timer;
   //PVCuda::CudaTimer * permute_preData_timer;
   //PVCuda::CudaTimer * permute_postGSyn_timer;
#endif
#endif

#ifdef PV_USE_OPENCL
   CLTimer * gpu_recvsyn_timer;
#endif
};

} // namespace PV

#endif /* HYPERLAYER_HPP_ */
