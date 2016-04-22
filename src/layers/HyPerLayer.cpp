/*
 * HyPerLayer.cpp
 *
 *  Created on: Jul 29, 2008
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
 *
 * DerivedLayer's constructors should only call the base class's default constructor (that only calls initialize_base).
 * This ensures that each class's initialize_base and initialize are only called once each, and that initialize_base
 * is called before initialize.
 *
 * initialize_base() should only set member variables to default values and member variable pointers to null.
 * initialize() should only store the constructor's arguments into member variables and read the hypercolumn's parameters,
 * storing the results in member variables.  Note that at the time the constructor (and therefore initialize) is called,
 * you cannot assume that any other layers and connections have been added to the HyPerCol.
 *
 * If you need to receive information from or send information to another object to fully initialize the layer,
 * override communicateInitInfo().  Be sure to call the base class's communicateInitInfo() within the derived class's
 * communicateInitInfo() method.
 *
 * If you have any buffers (e.g. conductances) that need to be allocated, do so by overriding allocateBuffers(),
 * which is called by HyPerLayer's allocateDataStructures().  Be sure to call the base class's allocateBuffers() within
 * the dervied class's allocateBuffers() method.  The reason for doing allocations here is that if a buffer is extended
 * and therefore depends on the value of the margin, it needs to wait until after the communicateInitInfo stage.
 * There may be some specialized layers that inherit their nx, ny, nf values from other layers, so that even for
 * restricted buffers it makes sense to wait until the allocateDataStructures stage to do the allocation.
 *
 */

#include <iostream>
#include <sstream>
#include "HyPerLayer.hpp"
#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../columns/HyPerCol.hpp"
#include "../connections/BaseConnection.hpp"
#include "../connections/TransposeConn.hpp"
#include "InitV.hpp"
#include "../io/fileio.hpp"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//void HyPerLayer_recv_post(
//      const int arborId,
//      const int nxRes, //num post neurons
//      const int nyRes,
//      const int nf,
//      const int nb, //Border of orig
//      const int nxp,
//      const int nyp,
//      const int nfp,
//      CL_MEM_GLOBAL long* startSourceExtBuf,
//      CL_MEM_GLOBAL float* preData,
//      CL_MEM_GLOBAL float* weights,
//      CL_MEM_GLOBAL float* postGSyn,
//      CL_MEM_GLOBAL int* patch2datalookuptable,
//      const int gsynAccumType,
//      const int sy,
//      const int syp,
//      const int numPerStride,
//      const float dt_factor,
//      const int sharedWeights
//);

#ifdef __cplusplus
}
#endif // __cplusplus

namespace PV {

///////
// This constructor is protected so that only derived classes can call it.
// It should be called as the normal method of object construction by
// derived classes.  It should NOT call any virtual methods
HyPerLayer::HyPerLayer() {
   initialize_base();
}

///////
// initialize_base should be called only by constructors.  It should not
// call any virtual methods, because polymorphism is not available when
// a base class constructor is inherited from a derived class constructor.
// In general, initialize_base should be used only to initialize member variables
// to safe values.
int HyPerLayer::initialize_base() {
   this->name = NULL;
   this->probes = NULL;
   this->nxScale = 1.0f;
   this->nyScale = 1.0f;
   this->numFeatures = 1;
   this->mirrorBCflag = 0;
   this->xmargin = 0;
   this->ymargin = 0;
   this->numProbes = 0;
   this->ioAppend = 0;
   this->numChannels = 2;
   this->clayer = NULL;
   this->GSyn = NULL;
   //this->labels = NULL;
   this->marginIndices = NULL;
   this->numMargin = 0;
   this->writeTime = 0;
   this->initialWriteTime = 0;
   this->triggerFlag = false; //Default to update every timestamp
   this->triggerLayer = NULL;
   this->triggerLayerName = NULL;
   this->triggerBehavior = NULL;
   this->triggerBehaviorType = NO_TRIGGER;
   this->triggerResetLayerName = NULL;
   this->initVObject = NULL;
   this->triggerOffset = 0;
   this->nextUpdateTime = 0;
   this->initializeFromCheckpointFlag = false;
   this->outputStateStream = NULL;
   
   this->lastUpdateTime = 0.0;
   this->phase = 0;

   this->initInfoCommunicatedFlag = false;
   this->dataStructuresAllocatedFlag = false;
   this->initialValuesSetFlag = false;
   
   this->numSynchronizedMarginWidthLayers = 0;
   this->synchronizedMarginWidthLayers = NULL;
   
   dataType = PV_FLOAT;
   dataTypeString = NULL;

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
//   this->krUpdate = NULL;
   this->allocDeviceV = false;
   this->allocDeviceGSyn = false;
   this->allocDeviceActivity = false;
   this->allocDeviceDatastore= false;
   this->allocDeviceActiveIndices= false;
   this->d_V = NULL;
   this->d_GSyn = NULL;
   this->d_Activity = NULL;
   this->d_Datastore= NULL;
   this->d_ActiveIndices= NULL;
   this->d_numActive = NULL;
   this->updatedDeviceActivity = true; //Start off always updating activity
   this->updatedDeviceDatastore = true;
   this->updatedDeviceGSyn = true;
   this->recvGpu = false;
   this->updateGpu = false;
   this->krUpdate = NULL;
#endif // PV_USE_OPENCL

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   this->cudnn_GSyn = NULL;
   this->cudnn_Datastore= NULL;
#endif

   this->update_timer  = NULL;
   this->recvsyn_timer = NULL;
   this->publish_timer = NULL;
   this->timescale_timer = NULL;
   this->io_timer      = NULL;

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   this->gpu_recvsyn_timer = NULL;
   this->gpu_update_timer = NULL;
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   //this->permute_weights_timer = NULL;
   //this->permute_preData_timer = NULL;
   //this->permute_postGSyn_timer = NULL;
#endif


   this->thread_gSyn = NULL;
   this->recvConns.clear();

   return PV_SUCCESS;
}

///////
/// Classes derived from HyPerLayer should call HyPerLayer::initialize themselves
/// to take advantage of virtual methods.  Note that the HyPerLayer constructor
/// does not call initialize.  This way, HyPerLayer::initialize can call virtual
/// methods and the derived class's method will be the one that gets called.
int HyPerLayer::initialize(const char * name, HyPerCol * hc) {
   int status = BaseLayer::initialize(name, hc);
   if (status != PV_SUCCESS) { return status; }

   // Timers
   this->update_timer =  new Timer(getName(), "layer", "update ");
   this->recvsyn_timer = new Timer(getName(), "layer", "recvsyn");
   this->publish_timer = new Timer(getName(), "layer", "publish");
   this->timescale_timer = new Timer(getName(), "layer", "timescale");
   this->io_timer =      new Timer(getName(), "layer", "io     ");

#ifdef PV_USE_CUDA
   this->gpu_recvsyn_timer = new PVCuda::CudaTimer(getName(), "layer", "gpurecvsyn");
   this->gpu_recvsyn_timer->setStream(hc->getDevice()->getStream());
   this->gpu_update_timer = new PVCuda::CudaTimer(getName(), "layer", "gpuupdate");
   this->gpu_update_timer->setStream(hc->getDevice()->getStream());
#ifdef PV_USE_CUDNN
   //this->permute_weights_timer = new PVCuda::CudaTimer(getName(), "layer", "gpuWeightsPermutate");
   //this->permute_weights_timer->setStream(hc->getDevice()->getStream());
   //this->permute_preData_timer = new PVCuda::CudaTimer(getName(), "layer", "gpuPreDataPermutate");
   //this->permute_preData_timer->setStream(hc->getDevice()->getStream());
   //this->permute_postGSyn_timer = new PVCuda::CudaTimer(getName(), "layer", "gpuPostGSynPermutate");
   //this->permute_postGSyn_timer->setStream(hc->getDevice()->getStream());
#endif
#endif

#ifdef PV_USE_OPENCL
   this->gpu_recvsyn_timer = hc->getDevice()->createTimer(getName(), "layer", "gpurecvsyn");
   this->gpu_update_timer = hc->getDevice()->createTimer(getName(), "layer", "gpuupdate");
#endif

   PVParams * params = parent->parameters();

   status = ioParams(PARAMS_IO_READ);
   assert(status == PV_SUCCESS);

   writeTime = initialWriteTime;
   writeActivityCalls = 0;
   writeActivitySparseCalls = 0;
   numDelayLevels = 1; // If a connection has positive delay so that more delay levels are needed, numDelayLevels is increased when BaseConnection::communicateInitInfo calls increaseDelayLevels
   maxRate = 1000.0f/parent->getDeltaTime();

   initClayer();

   // must set ioAppend before addLayer is called (addLayer causes activity file to be opened using layerid)
   ioAppend = parent->getCheckpointReadFlag() ? 1 : 0;

   layerId = parent->addLayer(this);

   lastUpdateTime = parent->simulationTime();
   nextUpdateTime = lastUpdateTime + parent->getDeltaTime();
   // nextTriggerTime will be initialized in communicateInitInfo(), as it depends on triggerLayer

//#ifdef PV_USE_OPENCL
//   initUseGPUFlag();
//#endif

   return PV_SUCCESS;
}

int HyPerLayer::initClayer() {
   clayer = (PVLayer *) calloc(1UL, sizeof(PVLayer));
   int status = PV_SUCCESS;
   if (clayer==NULL) {
      fprintf(stderr, "HyPerLayer \"%s\" error in rank %d process: unable to allocate memory for Clayer.\n", name, parent->columnId());
      exit(EXIT_FAILURE);
   }

   PVLayerLoc * loc = &clayer->loc;
   setLayerLoc(loc, nxScale, nyScale, numFeatures, parent->getNBatch());
   assert(loc->halo.lt==0 && loc->halo.rt==0 && loc->halo.dn==0 && loc->halo.up==0);

   int nBatch = parent->getNBatch();

   clayer->numNeurons  = loc->nx * loc->ny * loc->nf;
   clayer->numExtended = clayer->numNeurons; // initially, margin is zero; it will be updated as needed during the communicateInitInfo stage.
   clayer->numNeuronsAllBatches  = nBatch * loc->nx * loc->ny * loc->nf;
   clayer->numExtendedAllBatches = clayer->numNeuronsAllBatches;

   double xScaled = -log2( (double) nxScale);
   double yScaled = -log2( (double) nyScale);

   int xScale = (int) nearbyint(xScaled);
   int yScale = (int) nearbyint(yScaled);

   clayer->xScale = xScale;
   clayer->yScale = yScale;

   // Other fields of clayer will be set in allocateClayerBuffers, or during updateState
   return status;
}

//#ifdef PV_USE_OPENCL
////This method checks for a parameter telling Petavision to GPU accellerate
////this layer
//void HyPerLayer::initUseGPUFlag() {
//   PVParams * params = parent->parameters();
//   assert(!params->presentAndNotBeenRead(name,"GPUAccelerate"));
//   copyDataStoreFlag=false;
//}
//
////this method sets up GPU related variables and calls the
////initializeThreadBuffers and initializeThreadKernels
//int HyPerLayer::initializeGPU() {
//   CLDevice * device = parent->getDevice();
//
//   //copyToDevice=false;
//   numWait = 0;
//   numEvents = getNumCLEvents();
//   evList = (cl_event *) malloc(numEvents*sizeof(cl_event));
//   assert(evList != NULL);
//
//   // TODO - fix to use device and layer parameters
//   if (device->id() == 1) {
//      nxl = 1;  nyl = 1;
//   }
//   else {
//      nxl = 16; nyl = 8;
//   }
//
//   const char * kernel_name = getKernelName();
//   initializeThreadBuffers(kernel_name);
//   initializeThreadKernels(kernel_name);
//
//   return PV_SUCCESS;
//}
//#endif

HyPerLayer::~HyPerLayer()
{
   //if (parent->columnId() == 0) {
   //   writeTimers(stdout);
   //}
   delete recvsyn_timer;  recvsyn_timer = NULL;
   delete update_timer;   update_timer  = NULL;
   delete publish_timer;  publish_timer = NULL;
   delete timescale_timer;  timescale_timer = NULL;
   delete io_timer;       io_timer      = NULL;
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   delete gpu_recvsyn_timer; gpu_recvsyn_timer = NULL;
   delete gpu_update_timer; gpu_update_timer = NULL;
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   //delete permute_weights_timer; permute_weights_timer= NULL;
   //delete permute_preData_timer; permute_preData_timer= NULL;
   //delete permute_postGSyn_timer; permute_postGSyn_timer= NULL;
#endif

   if (outputStateStream) { pvp_close_file(outputStateStream, parent->icCommunicator()); }

   delete initVObject; initVObject = NULL;
   freeClayer();
   freeChannels();

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   if(krUpdate){
      delete krUpdate;
      krUpdate= NULL;
   }
   if(d_V){
      delete d_V;
   }
   if(d_Activity){
      delete d_Activity;
   }
   if(d_Datastore){
      delete d_Datastore;
   }

//      delete clPrevTime;
//      delete clParams;
//
//
//      free(evList);
//      evList = NULL;
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   if(cudnn_Datastore){
      delete cudnn_Datastore;
   }
#endif

   //free(labels); labels = NULL;
   free(marginIndices); marginIndices = NULL;
   free(probes); // All probes are deleted by the HyPerCol, so probes[i] doesn't need to be deleted, only the array itself.

   free(synchronizedMarginWidthLayers);
   
   free(triggerLayerName); triggerLayerName = NULL;
   free(triggerBehavior); triggerBehavior = NULL;
   free(triggerResetLayerName); triggerResetLayerName = NULL;

   if(thread_gSyn){
      for(int i = 0; i < parent->getNumThreads(); i++){
         free(thread_gSyn[i]);
         thread_gSyn[i] = NULL;
      }
      free(thread_gSyn);
      thread_gSyn = NULL;
   }

}

template <typename T>
int HyPerLayer::freeBuffer(T ** buf) {
   free(*buf);
   *buf = NULL;
   return PV_SUCCESS;
}
// Declare the instantiations of allocateBuffer that occur in other .cpp files; otherwise you may get linker errors.
template int HyPerLayer::freeBuffer<pvdata_t>(pvdata_t ** buf);
template int HyPerLayer::freeBuffer<int>(int ** buf);

int HyPerLayer::freeRestrictedBuffer(pvdata_t ** buf) {
   return freeBuffer(buf);
}

int HyPerLayer::freeExtendedBuffer(pvdata_t ** buf) {
   return freeBuffer(buf);
}

int HyPerLayer::freeClayer() {
   pvcube_delete(clayer->activity);

   freeBuffer(&clayer->prevActivity);
   freeBuffer(&clayer->V);
   free(clayer); clayer = NULL;

   return PV_SUCCESS;
}

void HyPerLayer::freeChannels()
{

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   if (d_GSyn != NULL) {
      delete d_GSyn;
      d_GSyn = NULL;
   }
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   if (cudnn_GSyn != NULL) {
      delete cudnn_GSyn;
   }
#endif


   // GSyn gets allocated in allocateDataStructures, but only if numChannels>0.
   if (GSyn) {
      assert(numChannels>0);
      free(GSyn[0]);  // conductances allocated contiguously so frees all buffer storage
      free(GSyn);     // this frees the array pointers to separate conductance channels
      GSyn = NULL;
      numChannels = 0;
   }
}

//#ifdef PV_USE_OPENCL
//#endif

int HyPerLayer::allocateClayerBuffers() {
   int k;
   // clayer fields numNeurons, numExtended, loc, xScale, yScale, dx, dy, xOrigin, yOrigin were set in initClayer().
   assert(clayer);
   clayer->params = NULL;

   int status = PV_SUCCESS;

   int statusV = allocateV();                      if (statusV!=PV_SUCCESS) status = PV_FAILURE;
   int statusA = allocateActivity();               if (statusA!=PV_SUCCESS) status = PV_FAILURE;
   //int statusActIndices = allocateActiveIndices(); if (statusActIndices!=PV_SUCCESS) status = PV_FAILURE;
   int statusPrevAct = allocatePrevActivity();     if (statusPrevAct!=PV_SUCCESS) status = PV_FAILURE;
   for (k = 0; k < getNumExtendedAllBatches(); k++) {
      clayer->prevActivity[k] = -10*REFRACTORY_PERIOD;  // allow neuron to fire at time t==0
   }

   return PV_SUCCESS;
}

template <typename T>
int HyPerLayer::allocateBuffer(T ** buf, int bufsize, const char * bufname) {
   int status = PV_SUCCESS;
   *buf = (T *) calloc(bufsize, sizeof(T));
   if(*buf == NULL) {
      fprintf(stderr, "Layer \"%s\" error in rank %d process: unable to allocate memory for %s: %s.\n", name, parent->columnId(), bufname, strerror(errno));
      status = PV_FAILURE;
   }
   return status;
}
// Declare the instantiations of allocateBuffer that occur in other .cpp files; otherwise you may get linker errors.
template int HyPerLayer::allocateBuffer<pvdata_t>(pvdata_t ** buf, int bufsize, const char * bufname);
template int HyPerLayer::allocateBuffer<int>(int ** buf, int bufsize, const char * bufname);

int HyPerLayer::allocateRestrictedBuffer(pvdata_t ** buf, char const * bufname) {
   return allocateBuffer(buf, getNumNeuronsAllBatches(), bufname);
}

int HyPerLayer::allocateExtendedBuffer(pvdata_t ** buf, char const * bufname) {
   return allocateBuffer(buf, getNumExtendedAllBatches(), bufname);
}

int HyPerLayer::allocateV() {
   return allocateRestrictedBuffer(&clayer->V, "membrane potential V");
}

int HyPerLayer::allocateActivity() {
   clayer->activity = pvcube_new(&clayer->loc, getNumExtendedAllBatches());
   return clayer->activity!=NULL ? PV_SUCCESS : PV_FAILURE;
}

int HyPerLayer::allocatePrevActivity() {
   return allocateExtendedBuffer(&clayer->prevActivity, "time of previous activity");
}

int HyPerLayer::setLayerLoc(PVLayerLoc * layerLoc, float nxScale, float nyScale, int nf, int numBatches)
{
   int status = PV_SUCCESS;

   InterColComm * icComm = parent->icCommunicator();

   float nxglobalfloat = nxScale * parent->getNxGlobal();
   layerLoc->nxGlobal = (int) nearbyintf(nxglobalfloat);
   if (fabs(nxglobalfloat-layerLoc->nxGlobal)>0.0001) {
      if (parent->columnId()==0) {
         fprintf(stderr, "nxScale of layer \"%s\" is incompatible with size of column.\n", getName());
         fprintf(stderr, "Column nx %d multiplied by nxScale %f must be an integer.\n", parent->getNxGlobal(), nxScale);
      }
      status = PV_FAILURE;
   }

   float nyglobalfloat = nyScale * parent->getNyGlobal();
   layerLoc->nyGlobal = (int) nearbyintf(nyglobalfloat);
   if (fabs(nyglobalfloat-layerLoc->nyGlobal)>0.0001) {
      if (parent->columnId()==0) {
         fprintf(stderr, "nyScale of layer \"%s\" is incompatible with size of column.\n", getName());
         fprintf(stderr, "Column ny %d multiplied by nyScale %f must be an integer.\n", parent->getNyGlobal(), nyScale);
      }
      status = PV_FAILURE;
   }

   // partition input space based on the number of processor
   // columns and rows
   //

   if (layerLoc->nxGlobal % icComm->numCommColumns() != 0) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Size of HyPerLayer \"%s\" is not  compatible with the mpi configuration.\n", name);
         fprintf(stderr, "The layer has %d pixels horizontally, and there are %d mpi processes in a row, but %d does not divide %d.\n",
               layerLoc->nxGlobal, icComm->numCommColumns(), icComm->numCommColumns(), layerLoc->nxGlobal);
      }
      status = PV_FAILURE;
   }
   if (layerLoc->nyGlobal % icComm->numCommRows() != 0) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Size of HyPerLayer \"%s\" is not  compatible with the mpi configuration.\n", name);
         fprintf(stderr, "The layer has %d pixels vertically, and there are %d mpi processes in a column, but %d does not divide %d.\n",
               layerLoc->nyGlobal, icComm->numCommRows(), icComm->numCommRows(), layerLoc->nyGlobal);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(icComm->communicator()); // If there is an error, make sure that MPI doesn't kill the run before process 0 reports the error.
   if (status != PV_SUCCESS) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Exiting.\n");
      }
      exit(EXIT_FAILURE);
   }
   layerLoc->nx = layerLoc->nxGlobal / icComm->numCommColumns();
   layerLoc->ny = layerLoc->nyGlobal / icComm->numCommRows();
   assert(layerLoc->nxGlobal == layerLoc->nx * icComm->numCommColumns());
   assert(layerLoc->nyGlobal == layerLoc->ny * icComm->numCommRows());

   layerLoc->kx0 = layerLoc->nx * icComm->commColumn();
   layerLoc->ky0 = layerLoc->ny * icComm->commRow();

   layerLoc->nf = nf;

   layerLoc->nbatch = numBatches;

   layerLoc->kb0 = parent->commBatch() * numBatches;
   layerLoc->nbatchGlobal = parent->numCommBatches() * numBatches;

   // halo is set in calls to updateClayerMargin
   layerLoc->halo.lt = 0; // margin;
   layerLoc->halo.rt = 0; // margin;
   layerLoc->halo.dn = 0; // margin;
   layerLoc->halo.up = 0; // margin;

   return 0;
}

void HyPerLayer::calcNumExtended() {
   PVLayerLoc const * loc = getLayerLoc();
   clayer->numExtended = (loc->nx+loc->halo.lt+loc->halo.rt)*(loc->ny+loc->halo.dn+loc->halo.up)*loc->nf;
   clayer->numExtendedAllBatches = clayer->numExtended * loc->nbatch;
}

int HyPerLayer::allocateBuffers() {
   // allocate memory for input buffers.  For HyPerLayer, allocates GSyn
   // virtual so that subclasses can initialize additional buffers if needed.
   // Typically an overriding allocateBuffers should call HyPerLayer::allocateBuffers
   // Specialized subclasses that don't use GSyn (e.g. CloneVLayer) should override
   // allocateGSyn to do nothing.
   
   return allocateGSyn();
}

int HyPerLayer::allocateGSyn() {
   int status = PV_SUCCESS;
   GSyn = NULL;
   if (numChannels > 0) {
      GSyn = (pvdata_t **) malloc(numChannels*sizeof(pvdata_t *));
      if(GSyn == NULL) {
         status = PV_FAILURE;
         return status;
      }

      GSyn[0] = (pvdata_t *) calloc(getNumNeuronsAllBatches()*numChannels, sizeof(pvdata_t));
      // All channels allocated at once and contiguously.  resetGSynBuffers_HyPerLayer() assumes this is true, to make it easier to port to GPU.
      if(GSyn[0] == NULL) {
         status = PV_FAILURE;
         return status;
      }

      for (int m = 1; m < numChannels; m++) {
         GSyn[m] = GSyn[0] + m * getNumNeuronsAllBatches();
      }
   }

   return status;
}

int HyPerLayer::initializeState() {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();

   if (parent->getCheckpointReadFlag()) {
      double checkTime = parent->simulationTime();
      checkpointRead(parent->getCheckpointReadDir(), &checkTime);
   }
   else if (parent->getInitializeFromCheckpointDir() && parent->getInitializeFromCheckpointDir()[0]) {
      assert(!params->presentAndNotBeenRead(name, "initializeFromCheckpointFlag"));
      if (initializeFromCheckpointFlag) {
         status = readStateFromCheckpoint(parent->getInitializeFromCheckpointDir(), NULL);
      }
   }
   else {
      status = setInitialValues();
   }
#ifdef PV_USE_CUDA
   copyInitialStateToGPU();
#endif // PV_USE_CUDA
   return status;
}

#ifdef PV_USE_CUDA
int HyPerLayer::copyInitialStateToGPU() {
   if(updateGpu){
      float * h_V = getV();
      if (h_V != NULL) {
         PVCuda::CudaBuffer* d_V = getDeviceV();
         assert(d_V);
         d_V->copyToDevice(h_V);
      }

      PVCuda::CudaBuffer* d_activity = getDeviceActivity();
      assert(d_activity);
      float * h_activity = getCLayer()->activity->data;
      d_activity->copyToDevice(h_activity);
   }
   return PV_SUCCESS;
}

#endif // PV_USE_CUDA

int HyPerLayer::setInitialValues() {
   int status = PV_SUCCESS;
   status = initializeV();
   if (status == PV_SUCCESS) initializeActivity();
   return status;
}

int HyPerLayer::initializeV() {
   int status = PV_SUCCESS;
   if (initVObject != NULL) {
      status = initVObject->calcV(this);
      setActivity();
   }
   return status;
}

int HyPerLayer::initializeActivity() {
   int status = setActivity();
   return status;
}

int HyPerLayer::ioParams(enum ParamsIOFlag ioFlag)
{
   parent->ioParamsStartGroup(ioFlag, name);
   ioParamsFillGroup(ioFlag);
   parent->ioParamsFinishGroup(ioFlag);

   return PV_SUCCESS;
}

int HyPerLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   // Derived classes with new params behavior should override ioParamsFillGroup
   // and the overriding method should call the base class's ioParamsFillGroup.
   ioParam_nxScale(ioFlag);
   ioParam_nyScale(ioFlag);
   ioParam_nf(ioFlag);
   ioParam_phase(ioFlag);
   ioParam_mirrorBCflag(ioFlag);
   ioParam_valueBC(ioFlag);
   ioParam_initializeFromCheckpointFlag(ioFlag);
   ioParam_InitVType(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_triggerBehavior(ioFlag);
   ioParam_triggerResetLayerName(ioFlag);
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_sparseLayer(ioFlag);
   ioParam_writeSparseValues(ioFlag);

   // GPU-specific parameter.  If not using GPUs, this flag
   // can be set to false or left out, but it is an error
   // to set updateGpu to true if compiling without GPUs.
   ioParam_updateGpu(ioFlag);

   ioParam_dataType(ioFlag);
   return PV_SUCCESS;
}

void HyPerLayer::ioParam_dataType(enum ParamsIOFlag ioFlag) {
   this->getParent()->ioParamString(ioFlag, this->getName(), "dataType", &dataTypeString, NULL, false/*warnIfAbsent*/);
   if(dataTypeString == NULL){
      //Default value
      dataType = PV_FLOAT;
      return;
   }
   if(!strcmp(dataTypeString, "float")){
      dataType = PV_FLOAT;
   }
   else if(!strcmp(dataTypeString, "int")){
      dataType = PV_INT;
   }
   else{
      std::cout << "BaseLayer " << name << " Error: dataType not recognized, can be \"float\" or \"int\"\n";
      exit(-1);
   }
}

void HyPerLayer::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   parent->ioParamValue(ioFlag, name, "updateGpu", &updateGpu, updateGpu, true/*warnIfAbsent*/);
#else // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   bool updateGpu = false;
   parent->ioParamValue(ioFlag, name, "updateGpu", &updateGpu, updateGpu, false/*warnIfAbsent*/);
   if (ioFlag==PARAMS_IO_READ && updateGpu) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: updateGpu is set to true, but PetaVision was compiled without GPU acceleration.\n",
               getKeyword(), getName());
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
}

void HyPerLayer::ioParam_nxScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nxScale", &nxScale, nxScale);
}

void HyPerLayer::ioParam_nyScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nyScale", &nyScale, nyScale);
}

void HyPerLayer::ioParam_nf(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nf", &numFeatures, numFeatures);
}

void HyPerLayer::ioParam_phase(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "phase", &phase, phase);
   if (ioFlag == PARAMS_IO_READ && phase<0) {
      if (parent->columnId()==0) fprintf(stderr, "Error in layer \"%s\": phase must be >= 0 (given value was %d).\n", name, phase);
      exit(EXIT_FAILURE);
   }
}

void HyPerLayer::ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "mirrorBCflag", &mirrorBCflag, mirrorBCflag);
}

void HyPerLayer::ioParam_valueBC(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "mirrorBCflag"));
   if (!mirrorBCflag) {
      parent->ioParamValue(ioFlag, name, "valueBC", &valueBC, (pvdata_t) 0);
   }
}

void HyPerLayer::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   assert(parent->getInitializeFromCheckpointDir());
   if (parent->getInitializeFromCheckpointDir() && parent->getInitializeFromCheckpointDir()[0]) {
      parent->ioParamValue(ioFlag, name, "initializeFromCheckpointFlag", &initializeFromCheckpointFlag, parent->getDefaultInitializeFromCheckpointFlag(), true/*warnIfAbsent*/);
   }
}

void HyPerLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initVObject = new InitV(parent, name);
      if( initVObject == NULL ) {
         fprintf(stderr, "%s \"%s\" error: unable to create InitV object\n", getKeyword(), name);
         abort();
      }
   }
   if (initVObject != NULL) {
      initVObject->ioParamsFillGroup(ioFlag);
   }
}

void HyPerLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "triggerLayerName", &triggerLayerName, NULL, false/*warnIfAbsent*/);
   if (ioFlag==PARAMS_IO_READ) {
      if (triggerLayerName && !strcmp(name, triggerLayerName)) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerLayerName cannot be the same as the name of the layer itself.\n",
                  getKeyword(), name);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      triggerFlag = (triggerLayerName!=NULL && triggerLayerName[0]!='\0');
   }
}

// triggerFlag was deprecated Aug 7, 2015.
// Setting triggerLayerName to a nonempty string has the effect of triggerFlag=true, and
// setting triggerLayerName to NULL or "" has the effect of triggerFlag=false.
// While triggerFlag is being deprecated, it is an error for triggerFlag to be false
// and triggerLayerName to be a nonempty string.
void HyPerLayer::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (ioFlag == PARAMS_IO_READ && parent->parameters()->present(name, "triggerFlag")) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Layer \"%s\" Warning: triggerFlag has been deprecated.\n", name);
      }
      bool flagFromParams = false;
      parent->ioParamValue(ioFlag, name, "triggerFlag", &flagFromParams, flagFromParams);
      if (flagFromParams != triggerFlag) {
         if (parent->columnId()==0) {
            fprintf(stderr, "Layer \"%s\" Error: triggerLayerName=", name);
            if (triggerLayerName) { fprintf(stderr, "\"%s\"", triggerLayerName); }
            else { fprintf(stderr, "NULL"); }
            fprintf(stderr, " implies triggerFlag=%s but triggerFlag was set in params to %s\n",
                  triggerFlag ? "true" : "false", flagFromParams ? "true" : "false");
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      else {
         if (parent->columnId()==0) {
            fprintf(stderr, "   If triggerLayerName is a nonempty string, triggering will be on;\n");
            fprintf(stderr, "   if triggerLayerName is empty or null, triggering will be off.\n");
         }
      }
   }
}

void HyPerLayer::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      parent->ioParamValue(ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
      if(triggerOffset < 0){
         fprintf(stderr, "%s \"%s\" error in rank %d process: TriggerOffset (%f) must be positive\n", getKeyword(), name, parent->columnId(), triggerOffset);
         exit(EXIT_FAILURE);
      }
   }
}
void HyPerLayer::ioParam_triggerBehavior(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      parent->ioParamString(ioFlag, name, "triggerBehavior", &triggerBehavior, "updateOnlyOnTrigger", true/*warnIfAbsent*/);
      if (triggerBehavior==NULL || !strcmp(triggerBehavior, "")) {
         free(triggerBehavior);
         triggerBehavior = strdup("updateOnlyOnTrigger");
         triggerBehaviorType = UPDATEONLY_TRIGGER;
      }
      else if (!strcmp(triggerBehavior, "updateOnlyOnTrigger")) {
         triggerBehaviorType = UPDATEONLY_TRIGGER;
      }
      else if (!strcmp(triggerBehavior, "resetStateOnTrigger")) {
         triggerBehaviorType = RESETSTATE_TRIGGER;
      }
      else if (!strcmp(triggerBehavior, "ignore")) {
         triggerBehaviorType = NO_TRIGGER;
      }
      else {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerBehavior=\"%s\" is unrecognized.\n",
                  getKeyword(), name, triggerBehavior);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   else { triggerBehaviorType = NO_TRIGGER; }
}

void HyPerLayer::ioParam_triggerResetLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerBehavior"));
      if (!strcmp(triggerBehavior, "resetStateOnTrigger")) {
         parent->ioParamStringRequired(ioFlag, name, "triggerResetLayerName", &triggerResetLayerName);
      }
   }
}

void HyPerLayer::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeStep", &writeStep, parent->getDeltaTime());
}

void HyPerLayer::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep>=0.0) {
      double start_time = parent->getStartTime();
      parent->ioParamValue(ioFlag, name, "initialWriteTime", &initialWriteTime, start_time);
      if (ioFlag == PARAMS_IO_READ && writeStep > 0.0 && initialWriteTime < start_time) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" warning: initialWriteTime is earlier than start time.  Adjusting initialWriteTime:\n",
                  getKeyword(), name);
         }
         while (initialWriteTime < start_time) {
            initialWriteTime += writeStep;
         }
         if (parent->columnId()==0) {
            fprintf(stderr, "    initialWriteTime adjusted to %f\n",initialWriteTime);
         }
      }
   }
}

void HyPerLayer::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && !parent->parameters()->present(name, "sparseLayer") && parent->parameters()->present(name, "writeSparseActivity")){
      parent->ioParamValue(ioFlag, name, "writeSparseActivity", &sparseLayer, false);
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: writeSparseActivity is deprecated.  Use sparseLayer instead.\n");
      }
      return;
   }
   // writeSparseActivity was deprecated Nov 4, 2014
   // When support for writeSparseActivity is removed entirely, remove the above if-statement and keep the ioParamValue call below.
   parent->ioParamValue(ioFlag, name, "sparseLayer", &sparseLayer, false);
}

void HyPerLayer::ioParam_writeSparseValues(enum ParamsIOFlag ioFlag) {
   // writeSparseActivity was deprecated Nov 4, 2014
   if(!parent->parameters()->present(name, "sparseLayer")){
      assert(!parent->parameters()->presentAndNotBeenRead(name, "writeSparseActivity"));
   }
   else{
      assert(!parent->parameters()->presentAndNotBeenRead(name, "sparseLayer"));
   }
   if (sparseLayer)
      parent->ioParamValue(ioFlag, name, "writeSparseValues", &writeSparseValues, true/*default value*/);
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

int HyPerLayer::allocateUpdateKernel(){
   std::cout << "Layer " << name << " of type " << getKeyword() << " does not support updating on gpus yet\n"; 
   exit(-1);
   return -1;
}

/**
 * Allocate GPU buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int HyPerLayer::allocateDeviceBuffers()
{
   int status = 0;

   
   const size_t size    = getNumNeuronsAllBatches()  * sizeof(float);
   const size_t size_ex = getNumExtendedAllBatches() * sizeof(float);

#ifdef PV_USE_OPENCL
   CLDevice * device = parent->getDevice();
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaDevice * device = parent->getDevice();
#endif 

   //Allocate based on which flags are set
   if(allocDeviceV){
#ifdef PV_USE_OPENCL
      d_V = device->createBuffer(CL_MEM_READ_WRITE, size, NULL);
#endif
#ifdef PV_USE_CUDA
      d_V = device->createBuffer(size);
#endif 
   }

   if(allocDeviceDatastore){
#ifdef PV_USE_OPENCL
      d_Datastore= device->createBuffer(CL_MEM_READ_ONLY, size_ex, NULL);
#endif
#ifdef PV_USE_CUDA
      d_Datastore= device->createBuffer(size_ex);
#endif 
      assert(d_Datastore);
#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
      cudnn_Datastore = device->createBuffer(size_ex);
      assert(cudnn_Datastore);
#endif
   }

   if(allocDeviceActiveIndices){
#ifdef PV_USE_OPENCL
      d_numActive = device->createBuffer(CL_MEM_READ_ONLY, parent->getNBatch() * sizeof(long), NULL);
      d_ActiveIndices = device->createBuffer(CL_MEM_READ_ONLY, size_ex, NULL);
#endif
#ifdef PV_USE_CUDA
      d_numActive = device->createBuffer(parent->getNBatch() * sizeof(long));
      d_ActiveIndices= device->createBuffer(size_ex);
#endif 
      assert(d_ActiveIndices);
   }

   if(allocDeviceActivity){
#ifdef PV_USE_OPENCL
      d_Activity = device->createBuffer(CL_MEM_READ_ONLY, size_ex, NULL);
#endif
#ifdef PV_USE_CUDA
      d_Activity = device->createBuffer(size_ex);
#endif 
   }

   //d_GSyn is the entire gsyn buffer. cudnn_GSyn is only one gsyn channel
   if(allocDeviceGSyn){
#ifdef PV_USE_OPENCL
      d_GSyn = device->createBuffer(CL_MEM_READ_WRITE, size * numChannels, NULL);
#endif
#ifdef PV_USE_CUDA
      d_GSyn = device->createBuffer(size * numChannels);
      assert(d_GSyn);
#ifdef PV_USE_CUDNN
      cudnn_GSyn = device->createBuffer(size);
#endif
#endif 
   }

   return status;
}

#endif

int HyPerLayer::communicateInitInfo()
{
   // HyPerLayers need to tell the parent HyPerCol how many random number
   // seeds they need.  At the start of HyPerCol::run, the parent HyPerCol
   // calls each layer's communicateInitInfo() sequentially in a repeatable order
   // (probably the order the layers appear in the params file) to make sure
   // that the same runs use the same RNG seeds in the same way.
   //
   // If any other object in the column needs the layer to have a certain minimum
   // margin width (e.g. a HyPerConn with patch size bigger than one), it should
   // call the layer's requireMarginWidth() method during its communicateInitInfo
   // stage.
   //
   // Since all communicateInitInfo() methods are called before any allocateDataStructures()
   // methods, HyPerLayer knows its marginWidth before it has to allocate
   // anything.  So the margin width does not have to be specified in params.
   if(triggerFlag){
      triggerLayer = parent->getLayerFromName(triggerLayerName);
      if (triggerLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerLayerName \"%s\" is not a layer in the HyPerCol.\n",
                  getKeyword(), name, triggerLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      nextTriggerTime = triggerLayer->getNextUpdateTime();
      if (triggerBehaviorType==RESETSTATE_TRIGGER) {
         char const * resetLayerName = NULL; // Will point to name of actual resetLayer, whether triggerResetLayerName is blank (in which case resetLayerName==triggerLayerName) or not
         if (triggerResetLayerName==NULL || triggerResetLayerName[0]=='\0') {
            resetLayerName = triggerLayerName;
            triggerResetLayer = triggerLayer;
         }
         else {
            resetLayerName = triggerResetLayerName;
            triggerResetLayer = parent->getLayerFromName(triggerResetLayerName);
            if (triggerResetLayer==NULL) {
               if (parent->columnId()==0) {
                  fprintf(stderr, "%s \"%s\" error: triggerResetLayerName \"%s\" is not a layer in the HyPerCol.\n",
                        getKeyword(), name, triggerResetLayerName);
               }
               MPI_Barrier(parent->icCommunicator()->communicator());
               exit(EXIT_FAILURE);
            }
         }
         // Check that triggerResetLayer and this layer have the same (restricted) dimensions.
         // Do we need to postpone until triggerResetLayer has finished its communicateInitInfo?
         PVLayerLoc const * triggerLoc = triggerResetLayer->getLayerLoc();
         PVLayerLoc const * localLoc = this->getLayerLoc();
         if (triggerLoc->nxGlobal != localLoc->nxGlobal || triggerLoc->nyGlobal != localLoc->nyGlobal || triggerLoc->nf != localLoc->nf) {
            if (parent->columnId()==0) {
               fprintf(stderr, "%s \"%s\" error: triggerResetLayer \"%s\" has incompatible dimensions.\n",
                     getKeyword(), name, resetLayerName);
               fprintf(stderr, "    \"%s\" is %d-by-%d-by-%d and \"%s\" is %d-by-%d-by-%d.\n",
                     name, localLoc->nxGlobal, localLoc->nyGlobal, localLoc->nf,
                     resetLayerName, triggerLoc->nxGlobal, triggerLoc->nyGlobal, triggerLoc->nf);
            }
         }
      }
   }

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   //Here, the connection tells all participating recev layers to allocate memory on gpu
   //if receive from gpu is set. These buffers should be set in allocate
   if(updateGpu){
      this->setAllocDeviceGSyn();
      this->setAllocDeviceV();
      this->setAllocDeviceActivity();
   }
#endif

   int status = PV_SUCCESS;

   return status;
}

char const * HyPerLayer::getOutputStatePath() {
   return outputStateStream ? outputStateStream->name : NULL;
}

int HyPerLayer::flushOutputStateStream() {
    int status = 0;
    if (outputStateStream && outputStateStream->fp) {
        status = fflush(outputStateStream->fp);
    }
    else {
        status = EOF;
        errno = EBADF;
    }
    return status;
}

int HyPerLayer::openOutputStateFile() {
   if (writeStep<0) { ioAppend = false; return PV_SUCCESS; }

   // If the communicator's batchwidth is greater than one, each local communicator creates an outputState file.
   // To prevent filename collisions, the global rank is inserted into the filename, just before the ".pvp" extension.
   // If the batchwidth is one, however, there is no need to insert the global rank.
   char appendCommBatchIdx[32];
   int numCommBatches = parent->icCommunicator()->numCommBatches();
   if (numCommBatches != 1) {
      int sz = snprintf(appendCommBatchIdx, 32, "_%d", parent->commBatch());
      if (sz >= 32) {
         fflush(stdout);
         fprintf(stderr, "%s \"%s\": Unable to create file name for outputState file: comm batch index %d is too long.\n",
               getKeyword(), name, parent->commBatch());
         exit(EXIT_FAILURE);
      }
   }
   else { // numCommBatches is one; insert the empty string instead.
      appendCommBatchIdx[0] = 0; // appendCommBatchIdx is the empty string
   }
   char filename[PV_PATH_MAX];
   char posFilename[PV_PATH_MAX];
   int sz;
   switch( parent->includeLayerName() ) {
   case 0:
      sz = snprintf(filename, PV_PATH_MAX, "%s/a%d%s.pvp", parent->getOutputPath(), layerId, appendCommBatchIdx);
      break;
   case 1:
      sz = snprintf(filename, PV_PATH_MAX, "%s/a%d_%s%s.pvp", parent->getOutputPath(), layerId, name, appendCommBatchIdx);
      break;
   case 2:
      sz = snprintf(filename, PV_PATH_MAX, "%s/%s%s.pvp", parent->getOutputPath(), name, appendCommBatchIdx);
      break;
   default:
      assert(0);
      break;
   }
   if (sz >= PV_PATH_MAX) {
      fflush(stdout);
      fprintf(stderr, "%s \"%s\": Unable to create file name for outputState file: file name with comm batch index %d is too long.\n",
            getKeyword(), name, parent->commBatch());
      exit(EXIT_FAILURE);
   }

   // initialize writeActivityCalls and writeSparseActivityCalls
   // only the root process needs these member variables so we don't need to do any MPI.
   int rootproc = 0;
   if (ioAppend && parent->columnId()==rootproc) {
      struct stat statbuffer;
      int filestatus = stat(filename, &statbuffer);
      if (filestatus == 0) {
         if (statbuffer.st_size==(off_t) 0)
         {
            ioAppend = false;
         }
      }
      else {
         if (errno==ENOENT) {
            ioAppend = false;
         }
         else {
            fprintf(stderr, "HyPerLayer::initializeLayerId error: stat \"%s\": %s\n", filename, strerror(errno));
            abort();
         }
      }
   }
   if (ioAppend && parent->columnId()==rootproc) {
      PV_Stream * pvstream = PV_fopen(filename,"r",false/*verifyWrites*/);
      if (pvstream) {
         int params[NUM_BIN_PARAMS];
         int numread = PV_fread(params, sizeof(int), NUM_BIN_PARAMS, pvstream);
         if (numread==NUM_BIN_PARAMS) {
            if (sparseLayer) {
               writeActivitySparseCalls = params[INDEX_NBANDS];
            }
            else {
               writeActivityCalls = params[INDEX_NBANDS];
            }
         }
         PV_fclose(pvstream);
      }
      else {
         ioAppend = false;
      }
   }
   InterColComm * icComm = parent->icCommunicator();
   MPI_Bcast(&ioAppend, 1, MPI_INT, 0/*root*/, icComm->communicator());
   outputStateStream = pvp_open_write_file(filename, icComm, ioAppend);
   return PV_SUCCESS;
}

void HyPerLayer::synchronizeMarginWidth(HyPerLayer * layer) {
   if (layer==this) { return; }
   assert(layer->getLayerLoc()!=NULL && this->getLayerLoc()!=NULL);
   HyPerLayer ** newSynchronizedMarginWidthLayers = (HyPerLayer **) calloc(numSynchronizedMarginWidthLayers+1, sizeof(HyPerLayer *));
   assert(newSynchronizedMarginWidthLayers);
   if (numSynchronizedMarginWidthLayers>0) {
      for (int k=0; k<numSynchronizedMarginWidthLayers; k++) {
         newSynchronizedMarginWidthLayers[k] = synchronizedMarginWidthLayers[k];
      }
      free(synchronizedMarginWidthLayers);
   }
   else {
      assert(synchronizedMarginWidthLayers==NULL);
   }
   synchronizedMarginWidthLayers = newSynchronizedMarginWidthLayers;
   synchronizedMarginWidthLayers[numSynchronizedMarginWidthLayers] = layer;
   numSynchronizedMarginWidthLayers++;

   equalizeMargins(this, layer);

   return;
}

int HyPerLayer::equalizeMargins(HyPerLayer * layer1, HyPerLayer * layer2) {
   int border1, border2, maxborder, result;
   int status = PV_SUCCESS;

   border1 = layer1->getLayerLoc()->halo.lt;
   border2 = layer2->getLayerLoc()->halo.lt;
   maxborder = border1 > border2 ? border1 : border2;
   layer1->requireMarginWidth(maxborder, &result, 'x');
   if (result != maxborder) { status = PV_FAILURE; }
   layer2->requireMarginWidth(maxborder, &result, 'x');
   if (result != maxborder) { status = PV_FAILURE; }
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Error in rank %d process: unable to synchronize x-margin widths of layers \"%s\" and \"%s\" to %d\n", layer1->getParent()->columnId(), layer1->getName(), layer2->getName(), maxborder);;
      exit(EXIT_FAILURE);
   }
   assert(layer1->getLayerLoc()->halo.lt == layer2->getLayerLoc()->halo.lt &&
          layer1->getLayerLoc()->halo.rt == layer2->getLayerLoc()->halo.rt &&
          layer1->getLayerLoc()->halo.lt == layer1->getLayerLoc()->halo.rt &&
          layer1->getLayerLoc()->halo.lt == maxborder);

   border1 = layer1->getLayerLoc()->halo.dn;
   border2 = layer2->getLayerLoc()->halo.dn;
   maxborder = border1 > border2 ? border1 : border2;
   layer1->requireMarginWidth(maxborder, &result, 'y');
   if (result != maxborder) { status = PV_FAILURE; }
   layer2->requireMarginWidth(maxborder, &result, 'y');
   if (result != maxborder) { status = PV_FAILURE; }
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Error in rank %d process: unable to synchronize y-margin widths of layers \"%s\" and \"%s\" to %d\n", layer1->getParent()->columnId(), layer1->getName(), layer2->getName(), maxborder);;
      exit(EXIT_FAILURE);
   }
   assert(layer1->getLayerLoc()->halo.dn == layer2->getLayerLoc()->halo.dn &&
          layer1->getLayerLoc()->halo.up == layer2->getLayerLoc()->halo.up &&
          layer1->getLayerLoc()->halo.dn == layer1->getLayerLoc()->halo.up &&
          layer1->getLayerLoc()->halo.dn == maxborder);
   return status;
}

int HyPerLayer::allocateDataStructures()
{
   std::cout.flush();
   // Once initialize and communicateInitInfo have been called, HyPerLayer has the
   // information it needs to allocate the membrane potential buffer V, the
   // activity buffer activity->data, and the data store.
   int status = PV_SUCCESS;

   //Doing this check here, since trigger layers are being set up in communicateInitInfo
   //If the magnitude of the trigger offset is bigger than the delta update time, then error
   if(triggerFlag){
      double deltaUpdateTime = getDeltaUpdateTime();
      if(deltaUpdateTime != -1 && triggerOffset >= deltaUpdateTime){ 
         fprintf(stderr, "%s \"%s\" error in rank %d process: TriggerOffset (%f) must be lower than the change in update time (%f) \n", getKeyword(), name, parent->columnId(), triggerOffset, deltaUpdateTime);
         exit(EXIT_FAILURE);
      }
   }

   allocateClayerBuffers();

   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   PVHalo const * halo = &loc->halo;

   // If not mirroring, fill the boundaries with the value in the valueBC param
   if (!useMirrorBCs() && getValueBC()!=0.0f) {
      int idx = 0;
      for(int batch=0; batch < loc->nbatch; batch++){
         for (int b=0; b<halo->up; b++) {
            for(int k=0; k<(nx+halo->lt+halo->rt)*nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
         }
         for (int y=0; y<ny; y++) {
            for(int k=0; k<halo->lt*nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
            idx += nx;
            for(int k=0; k<halo->rt*nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
         }
         for (int b=0; b<halo->dn; b++) {
            for(int k=0; k<(nx+halo->lt+halo->rt)*nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
         }
      }
      assert(idx==getNumExtendedAllBatches());
   }

   // allocate storage for the input conductance arrays
   status = allocateBuffers();
   assert(status == PV_SUCCESS);

   //Labels deprecated 6/16/15
   //// labels are not extended
   //labels = (int *) calloc(getNumNeurons(), sizeof(int));
   //if (labels==NULL) {
   //   fprintf(stderr, "HyPerLayer \"%s\" error: rank %d unable to allocate memory for labels.\n", name, parent->columnId());
   //   exit(EXIT_FAILURE);
   //}

   //Allocate temp buffers if needed, 1 for each thread
   if(parent->getNumThreads() > 1){
      thread_gSyn = (pvdata_t**) malloc(sizeof(pvdata_t*) * parent->getNumThreads());
      assert(thread_gSyn);

      //Assign thread_gSyn to different points of tempMem
      for(int i = 0; i < parent->getNumThreads(); i++){
         pvdata_t* tempMem = (pvdata_t*) malloc(sizeof(pvdata_t) * getNumNeuronsAllBatches());
         if(!tempMem){
            fprintf(stderr, "HyPerLayer \"%s\" error: rank %d unable to allocate %zu memory for thread_gSyn: %s\n", name, parent->columnId(), sizeof(pvdata_t) * getNumNeuronsAllBatches(), strerror(errno));
            exit(EXIT_FAILURE);
         }
         thread_gSyn[i] = tempMem;
      }

   }

   //Allocate opencl stuff on gpu if set
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   status = allocateDeviceBuffers();
   //Allocate receive from post kernel
   if(status == 0){
      status = PV_SUCCESS;
   }
   else{
      fprintf(stderr, "Connection \"%s\" unable to allocate device memory in rank %d process: %s\n", getName(), getParent()->columnId(), strerror(errno));
      exit(PV_FAILURE);
   }
   if(updateGpu){
      //This function needs to be overwritten as needed on a subclass basis
      status = allocateUpdateKernel();
      if(status == 0){
         status = PV_SUCCESS;
      }
   }
#endif

   //Make a data structure that stores the connections (in order of execution) this layer needs to recv from
   //CPU connections must run first to avoid race conditions
   int numConnections = parent->numberOfConnections();
   for(int c=0; c<numConnections; c++){
      BaseConnection * baseConn = parent->getConnection(c);
      HyPerConn * conn = dynamic_cast<HyPerConn *>(baseConn);
      if(conn->postSynapticLayer()!=this) continue;
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
      //If not recv from gpu, execute first
      if(!conn->getReceiveGpu()){
         recvConns.insert(recvConns.begin(), conn);
      }
      //Otherwise, add to the back. If no gpus at all, just add to back
      else
#endif
      {
         recvConns.push_back(conn);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
         //If it is receiving from gpu, set layer flag as such
         recvGpu = true;
#endif
      }
   }

   if (status == PV_SUCCESS) {
      status = openOutputStateFile();
   }

   return status;
}

/*
 * Call this routine to increase the number of levels in the data store ring buffer.
 * Calls to this routine after the data store has been initialized will have no effect.
 * The routine returns the new value of numDelayLevels
 */
int HyPerLayer::increaseDelayLevels(int neededDelay) {
   if( numDelayLevels < neededDelay+1 ) numDelayLevels = neededDelay+1;
   if( numDelayLevels > MAX_F_DELAY ) numDelayLevels = MAX_F_DELAY;
   return numDelayLevels;
}

int HyPerLayer::requireMarginWidth(int marginWidthNeeded, int * marginWidthResult, char axis) {
   // TODO: Is there a good way to handle x- and y-axis margins without so much duplication of code?
   // Navigating through the halo makes it difficult to combine cases.
   PVLayerLoc * loc = &clayer->loc;
   PVHalo * halo = &loc->halo;
   switch (axis) {
   case 'x':
      *marginWidthResult = xmargin;
      if (xmargin < marginWidthNeeded) {
         assert(clayer);
         if (parent->columnId()==0) {
            printf("Layer \"%s\": adjusting x-margin width from %d to %d\n", name, xmargin, marginWidthNeeded);
         }
         xmargin = marginWidthNeeded;
         halo->lt = xmargin;
         halo->rt = xmargin;
         calcNumExtended();
         assert(axis=='x' && getLayerLoc()->halo.lt==getLayerLoc()->halo.rt);
         *marginWidthResult = xmargin;
         if (synchronizedMarginWidthLayers != NULL) {
            for (int k=0; k<numSynchronizedMarginWidthLayers; k++) {
               HyPerLayer * l = synchronizedMarginWidthLayers[k];
               if (l->getLayerLoc()->halo.lt < marginWidthNeeded) {
                  synchronizedMarginWidthLayers[k]->requireMarginWidth(marginWidthNeeded, marginWidthResult, axis);
               }
               assert(l->getLayerLoc()->halo.lt == getLayerLoc()->halo.lt);
               assert(l->getLayerLoc()->halo.rt == getLayerLoc()->halo.rt);
            }
         }
      }
      break;
   case 'y':
      *marginWidthResult = ymargin;
      if (ymargin < marginWidthNeeded) {
         assert(clayer);
         if (parent->columnId()==0) {
            printf("Layer \"%s\": adjusting y-margin width from %d to %d\n", name, ymargin, marginWidthNeeded);
         }
         ymargin = marginWidthNeeded;
         halo->dn = ymargin;
         halo->up = ymargin;
         calcNumExtended();
         assert(axis=='y' && getLayerLoc()->halo.dn==getLayerLoc()->halo.up);
         *marginWidthResult = ymargin;
         if (synchronizedMarginWidthLayers != NULL) {
            for (int k=0; k<numSynchronizedMarginWidthLayers; k++) {
               HyPerLayer * l = synchronizedMarginWidthLayers[k];
               if (l->getLayerLoc()->halo.up < marginWidthNeeded) {
                  synchronizedMarginWidthLayers[k]->requireMarginWidth(marginWidthNeeded, marginWidthResult, axis);
               }
               assert(l->getLayerLoc()->halo.dn == getLayerLoc()->halo.dn);
               assert(l->getLayerLoc()->halo.up == getLayerLoc()->halo.up);
            }
         }
      }
      break;
   default:
      assert(0);
      break;
   }
   return PV_SUCCESS;
}

int HyPerLayer::requireChannel(int channelNeeded, int * numChannelsResult) {
   if (channelNeeded >= numChannels) {
      int numOldChannels = numChannels;
      numChannels = channelNeeded+1;
   }
   *numChannelsResult = numChannels;

   return PV_SUCCESS;
}

/**
 * Returns the activity data for the layer.  This data is in the
 * extended space (with margins).
 */
const pvdata_t * HyPerLayer::getLayerData(int delay)
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   return (pvdata_t *) store->buffer(0, delay);
}

/**
 * Copy cube data to the border region while applying boundary conditions
 *   - this implements mirror boundary conditions
 *   - assumes both input PVLayerCubes are of identical size and shape, typically the same struct
 */
int HyPerLayer::mirrorInteriorToBorder(int whichBorder, PVLayerCube * cube, PVLayerCube * border)
{
   assert( cube->numItems == border->numItems );
   assert( localDimensionsEqual(&cube->loc,&border->loc));
   int status = 0;
   switch (whichBorder) {
   case NORTHWEST:
      status = mirrorToNorthWest(border, cube); break;
   case NORTH:
      status = mirrorToNorth(border, cube); break;
   case NORTHEAST:
      status = mirrorToNorthEast(border, cube); break;
   case WEST:
      status = mirrorToWest(border, cube); break;
   case EAST:
      status = mirrorToEast(border, cube); break;
   case SOUTHWEST:
      status = mirrorToSouthWest(border, cube); break;
   case SOUTH:
      status = mirrorToSouth(border, cube); break;
   case SOUTHEAST:
      status = mirrorToSouthEast(border, cube); break;
   default:
      fprintf(stderr, "ERROR:HyPerLayer:copyToBorder: bad border index %d\n", whichBorder);
      status = -1;
      break;
   }
   return status;
}

int HyPerLayer::mirrorInteriorToBorder(PVLayerCube * cube, PVLayerCube * border)
{
   assert( cube->numItems == border->numItems );
   assert( localDimensionsEqual(&cube->loc,&border->loc));

   mirrorToNorthWest(border, cube);
   mirrorToNorth(border, cube);
   mirrorToNorthEast(border, cube);
   mirrorToWest(border, cube);
   mirrorToEast(border, cube);
   mirrorToSouthWest(border, cube);
   mirrorToSouth(border, cube);
   mirrorToSouthEast(border, cube);
   return 0;
}

bool HyPerLayer::needUpdate(double time, double dt){
   bool updateNeeded = false;
   // Check if the layer ever updates.
   if (getDeltaUpdateTime() < 0) {
      updateNeeded = false;
   }
   //Return true if the layer was updated this timestep as well
   else if(fabs(parent->simulationTime() - lastUpdateTime) < (dt/2)){
      updateNeeded = true;
   }
   else {
      //We want to check whether time==nextUpdateTime-triggerOffset,
      // but to account for roundoff errors, we check if it's within half the delta time
      updateNeeded = fabs(time - (nextUpdateTime - triggerOffset)) < (dt/2);
   }
   return updateNeeded;
}

int HyPerLayer::updateNextUpdateTime(){
   double deltaUpdateTime = getDeltaUpdateTime();
   assert(deltaUpdateTime != 0);
   if(deltaUpdateTime > 0){
      while(parent->simulationTime() >= nextUpdateTime){
         nextUpdateTime += deltaUpdateTime;
      }
   }
   return PV_SUCCESS;
}

double HyPerLayer::getDeltaUpdateTime(){
   if(triggerLayer != NULL && triggerBehaviorType == UPDATEONLY_TRIGGER){
      return getDeltaTriggerTime();
   }
   else{
      return parent->getDeltaTime();
   }
}

int HyPerLayer::updateNextTriggerTime() {
   if (triggerLayer==NULL) { return PV_SUCCESS; }
   double deltaTriggerTime = getDeltaTriggerTime();
   if (deltaTriggerTime > 0) {
      while(parent->simulationTime() >= nextTriggerTime) {
         nextTriggerTime += deltaTriggerTime;
      }
   }
   return PV_SUCCESS;
}

double HyPerLayer::getDeltaTriggerTime(){
   if(triggerLayer != NULL){
      return triggerLayer->getDeltaUpdateTime();
   }
   else{
      return -1;
   }
}

bool HyPerLayer::needReset(double timed, double dt) {
   bool resetNeeded = false;
   if (triggerLayer != NULL && triggerBehaviorType == RESETSTATE_TRIGGER) {
      resetNeeded = fabs(timed - (nextTriggerTime - triggerOffset)) < (dt/2);
   }
   return resetNeeded;
}

int HyPerLayer::updateStateWrapper(double timef, double dt){
   int status = PV_SUCCESS;
   if(needUpdate(timef, parent->getDeltaTime())){
      if (needReset(timef, dt)) {
         status = resetStateOnTrigger();
         updateNextTriggerTime();
      }
      status = callUpdateState(timef, dt);
      lastUpdateTime = parent->simulationTime();
   }
   //Because of the triggerOffset, we need to check if we need to update nextUpdateTime every time
   updateNextUpdateTime();
   return status;
}

int HyPerLayer::callUpdateState(double timed, double dt) {
   int status = PV_SUCCESS;
#ifdef PV_USE_OPENCL
   //If this current layer's gsyn is on the gpu, only move it back when doing update state or output state
   this->clFinishGSyn();
#endif
   update_timer->start();
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   if(updateGpu){
      gpu_update_timer->start();
      status = updateStateGpu(timed, dt);
      gpu_update_timer->stop();
   }
   else{
#endif
      status = updateState(timed, dt);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   }
   //Activity updated, set flag to true
   updatedDeviceActivity = true;
   updatedDeviceDatastore = true;
#endif
   update_timer->stop();
   return status;
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
//Multiple entry points into doUpdateStateGpu in case a layer overwrites updateState
int HyPerLayer::updateStateGpu(double timef, double dt){
   int status;
   pvdata_t * gSynHead = GSyn==NULL ? NULL : GSyn[0];
   assert(updateGpu);
   status = doUpdateStateGpu(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(),
         getNumChannels(), gSynHead);
   return status;
}
#endif

int HyPerLayer::updateState(double timef, double dt) {
   int status;
   pvdata_t * gSynHead = GSyn==NULL ? NULL : GSyn[0];

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   if(updateGpu){
      status = doUpdateStateGpu(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(),
            getNumChannels(), gSynHead);
   }
   else{
#endif
      status = doUpdateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(),
            getNumChannels(), gSynHead);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   }
#endif

   return status;
}

int HyPerLayer::resetStateOnTrigger() {
   assert(triggerResetLayer != NULL);
   pvpotentialdata_t * V = getV();
   if (V==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: triggerBehavior is \"resetStateOnTrigger\" but layer does not have a membrane potential.\n",
               getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   pvpotentialdata_t const * resetV = triggerResetLayer->getV();
   if (resetV!=NULL) {
      #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
      #endif // PV_USE_OPENMP_THREADS
      for (int k=0; k<getNumNeurons(); k++) {
         V[k] = resetV[k];
      }
   }
   else {
      pvadata_t const * resetA = triggerResetLayer->getActivity();
      PVLayerLoc const * loc = triggerResetLayer->getLayerLoc();
      PVHalo const * halo = &loc->halo;
      #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
      #endif // PV_USE_OPENMP_THREADS
      for (int k=0; k<getNumNeurons(); k++) {
         int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
         V[k] = resetA[kex];
      }
   }
   return setActivity();
}

int HyPerLayer::resetGSynBuffers(double timef, double dt) {
   int status = PV_SUCCESS;
   if (GSyn == NULL) return PV_SUCCESS;
   resetGSynBuffers_HyPerLayer(parent->getNBatch(), this->getNumNeurons(), getNumChannels(), GSyn[0]); // resetGSynBuffers();
   return status;
}


#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
int HyPerLayer::runUpdateKernel(){

#ifdef PV_USE_CUDA
   assert(updateGpu);
   if(updatedDeviceGSyn){
      copyAllGSynToDevice();
      updatedDeviceGSyn = false;
   }

   //V and Activity are write only buffers, so we don't need to do anything with them
   assert(krUpdate);
   //Run kernel
   krUpdate->run();
#endif

   return PV_SUCCESS;
}

int HyPerLayer::doUpdateStateGpu(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
   std::cout << "Update state for layer " << name << " is not implemented\n";
   exit(-1);
   return -1;
}
#endif

int HyPerLayer::doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead)
{
   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nbatch = loc->nbatch;
   int num_neurons = nx*ny*nf;
   if (num_channels == 1){
      applyGSyn_HyPerLayer1Channel(nbatch, num_neurons, V, gSynHead);
   }
   else{
      applyGSyn_HyPerLayer(nbatch, num_neurons, V, gSynHead);
   }
   setActivity_HyPerLayer(nbatch, num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);

   return PV_SUCCESS;
}

int HyPerLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_HyPerLayer(loc->nbatch, getNumNeurons(), clayer->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
}

int HyPerLayer::updateBorder(double time, double dt)
{
   int status = PV_SUCCESS;

//#ifdef PV_USE_OPENCL
//   // wait for memory to be copied from device
//   if (numWait > 0) {
//      status |= clWaitForEvents(numWait, evList);
//   }
//   for (int i = 0; i < numWait; i++) {
//      clReleaseEvent(evList[i]);
//   }
//   numWait = 0;
//
//   //   status |= clWaitForEvents(1, &evUpdate);
//   //   clReleaseEvent(evUpdate);
//#endif

   return status;
}

//Updates active indices for all levels (delays) here
int HyPerLayer::updateAllActiveIndices() {
   return parent->icCommunicator()->updateAllActiveIndices(this->getLayerId());
}
int HyPerLayer::updateActiveIndices() {
   return parent->icCommunicator()->updateActiveIndices(this->getLayerId());
}

int HyPerLayer::recvAllSynapticInput() {
   int status = PV_SUCCESS;
   //Only recvAllSynapticInput if we need an update
   if(needUpdate(parent->simulationTime(), parent->getDeltaTime())){
      bool switchGpu = false;
      //Start CPU timer here
      recvsyn_timer->start();

      for(std::vector<BaseConnection*>::iterator it = recvConns.begin(); it < recvConns.end(); it++){
         BaseConnection * baseConn = *it;
         HyPerConn * conn = dynamic_cast<HyPerConn *>(baseConn);
         assert(conn != NULL);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
         //Check if it's done with cpu connections
         if(!switchGpu && conn->getReceiveGpu()){
            //Copy GSyn over to GPU
            copyAllGSynToDevice();
#ifdef PV_USE_CUDA
            //Start gpu timer
            gpu_recvsyn_timer->start();
#endif
            switchGpu = true;
         }
#endif
         conn->deliver();
      }
#ifdef PV_USE_CUDA
      if(switchGpu){
         //Stop timer
         gpu_recvsyn_timer->stop();
      }
#endif
      recvsyn_timer->stop();
   }
   return status;
}


#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
float HyPerLayer::addGpuTimers(){
   float time = 0;
   bool updateNeeded = needUpdate(parent->simulationTime(), parent->getDeltaTime());
   if(recvGpu && updateNeeded){
      time += gpu_recvsyn_timer->accumulateTime();
   }
   if(updateGpu && updateNeeded){
      time += gpu_update_timer->accumulateTime();
   }
   return time;
}

void HyPerLayer::syncGpu(){
   if(recvGpu || updateGpu){
#ifdef PV_USE_CUDA
      parent->getDevice()->syncDevice();
#endif
#ifdef PV_USE_OPENCL
      parent->getDevice()->syncDevice();
#endif
   }
}
#endif

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
void HyPerLayer::copyAllGSynToDevice(){
   if(recvGpu || updateGpu){
      //Copy it to device
      //Allocated as a big chunk, this should work
      float * h_postGSyn = GSyn[0];
#ifdef PV_USE_OPENCL
      CLBuffer * d_postGSyn = this->getDeviceGSyn();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_postGSyn = this->getDeviceGSyn();
#endif
      assert(d_postGSyn);
      d_postGSyn->copyToDevice(h_postGSyn);
   }
}

void HyPerLayer::copyAllGSynFromDevice(){
   //Only copy if recving
   if(recvGpu){
      //Allocated as a big chunk, this should work
      float * h_postGSyn = GSyn[0];
#ifdef PV_USE_OPENCL
      CLBuffer * d_postGSyn = this->getDeviceGSyn();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_postGSyn = this->getDeviceGSyn();
#endif
      assert(d_postGSyn);
      d_postGSyn->copyFromDevice(h_postGSyn);
   }
}

void HyPerLayer::copyAllVFromDevice(){
   //Only copy if updating
   if(updateGpu){
      //Allocated as a big chunk, this should work
      float * h_V = getV();
#ifdef PV_USE_OPENCL
      CLBuffer * d_V = this->getDeviceV();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_V= this->getDeviceV();
#endif
      assert(d_V);
      d_V->copyFromDevice(h_V);
   }
}

void HyPerLayer::copyAllActivityFromDevice(){
   //Only copy if updating
   if(updateGpu){
      //Allocated as a big chunk, this should work
      float * h_activity = getCLayer()->activity->data;
#ifdef PV_USE_OPENCL
      CLBuffer * d_activity = this->getDeviceActivity();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_activity= this->getDeviceActivity();
#endif
      assert(d_activity);
      d_activity->copyFromDevice(h_activity);
   }
}

#endif

int HyPerLayer::publish(InterColComm* comm, double time)
{
   publish_timer->start();

   bool mirroring = useMirrorBCs();
   mirroring = mirroring ?
         (getLastUpdateTime() >= getParent()->simulationTime()) :
         false;
   if ( mirroring) {
      for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
         mirrorInteriorToBorder(borderId, clayer->activity, clayer->activity);
      }
   }
   
   int status = comm->publish(this, clayer->activity);
//#ifdef PV_USE_OPENCL
//   if(copyDataStoreFlag) {
//      status |= copyDataStoreCLBuffer();
//      //status |= getLayerDataStoreCLBuffer()->copyToDevice(evCopyDataStore);
//      //numWait += 1;
//   }
//#endif

   publish_timer->stop();
   return status;
}

int HyPerLayer::waitOnPublish(InterColComm* comm)
{
   publish_timer->start();

   // wait for MPI border transfers to complete
   //
   int status = comm->wait(getLayerId());

   publish_timer->stop();
   return status;
}

/******************************************************************
 * FileIO 
 *****************************************************************/

/* Inserts a new probe into an array of LayerProbes.
 *
 *
 *
 */
int HyPerLayer::insertProbe(LayerProbe * p)
{
   if(p->getTargetLayer() != this) {
      fprintf(stderr, "HyPerLayer \"%s\": insertProbe called with probe %p, whose targetLayer is not this layer.  Probe was not inserted.\n", name, p);
      return numProbes;
   }
   for( int i=0; i<numProbes; i++ ) {
      if( p == probes[i] ) {
         fprintf(stderr, "HyPerLayer \"%s\": insertProbe called with probe %p, which has already been inserted as probe %d.\n", name, p, i);
         return numProbes;
      }
   }

   // malloc'ing a new buffer, copying data over, and freeing the old buffer could be replaced by malloc
   LayerProbe ** tmp;
   tmp = (LayerProbe **) malloc((numProbes + 1) * sizeof(LayerProbe *));
   assert(tmp != NULL);

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   free(probes);

   probes = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerLayer::outputProbeParams() {
   int status = PV_SUCCESS;
   for (int p=0; p<numProbes; p++) {
      int status1 = probes[p]->ioParams(PARAMS_IO_WRITE);
      if (status1 != PV_SUCCESS) { status = PV_FAILURE; }
   }
   return status;
}

int HyPerLayer::outputState(double timef, bool last)
{


#ifdef PV_USE_OPENCL
   //Make sure all data is finished before this point
   clFinishGSyn();
   clFinishActivity();
#endif

   int status = PV_SUCCESS;

   io_timer->start();

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputStateWrapper(timef, parent->getDeltaTime());
   }


   if (timef >= (writeTime-(parent->getDeltaTime()/2)) && writeStep >= 0) {
      writeTime += writeStep;
      if (sparseLayer) {
         status = writeActivitySparse(timef, writeSparseValues);
      }
      else {
         status = writeActivity(timef);
      }
   }
   if (status!=PV_SUCCESS) {
      fprintf(stderr, "%s \"%s\": outputState failed on rank %d process.\n", getKeyword(), name, parent->columnId());
      exit(EXIT_FAILURE);
   }

   io_timer->stop();
   return status;
}

int HyPerLayer::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   // If timeptr is NULL, the timestamps in the pvp files are ignored.  If non-null, they are compared to the value of *timeptr and
   // a warning is issued if there is a discrepancy.
   int status = PV_SUCCESS;
   status = readActivityFromCheckpoint(cpDir, timeptr);
   status = readVFromCheckpoint(cpDir, timeptr);
   status = readDelaysFromCheckpoint(cpDir, timeptr);
   return status;
}

int HyPerLayer::readActivityFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_A.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &clayer->activity->data, 1, /*extended*/true, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
   assert(status==PV_SUCCESS);
   return status;
}

int HyPerLayer::readVFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = PV_SUCCESS;
   if (getV() != NULL) {
      char * filename = parent->pathInCheckpoint(cpDir, getName(), "_V.pvp");
      pvdata_t * V = getV();
      status = readBufferFile(filename, parent->icCommunicator(), timeptr, &V, 1, /*extended*/false, getLayerLoc());
      assert(status == PV_SUCCESS);
      free(filename);
   }
   return status;
}

int HyPerLayer::readDelaysFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_Delays.pvp");
   int status = readDataStoreFromFile(filename, parent->icCommunicator(), timeptr);
   assert(status == PV_SUCCESS);
   free(filename);
   return status;
}

int HyPerLayer::checkpointRead(const char * cpDir, double * timeptr) {
   int status = readStateFromCheckpoint(cpDir, timeptr);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "Layer \"%s\": rank %d process failed to read state from checkpoint directory \"%s\"\n", getName(), parent->columnId(), cpDir);
      exit(EXIT_FAILURE);
   }
   InterColComm * icComm = parent->icCommunicator();
   parent->readScalarFromFile(cpDir, getName(), "lastUpdateTime", &lastUpdateTime, parent->simulationTime()-parent->getDeltaTime());
   parent->readScalarFromFile(cpDir, getName(), "nextUpdateTime", &nextUpdateTime, parent->simulationTime()+parent->getDeltaTime());
   parent->readScalarFromFile(cpDir, getName(), "nextWrite", &writeTime, writeTime);

   if (ioAppend) {
      long activityfilepos = 0L;
      parent->readScalarFromFile(cpDir, getName(), "filepos", &activityfilepos);
      if (parent->columnId()==0 && outputStateStream) {
         if (PV_fseek(outputStateStream, activityfilepos, SEEK_SET) != 0) {
            fprintf(stderr, "HyPerLayer::checkpointRead error: unable to recover initial file position in activity file for layer %s\n", name);
            abort();
         }
      }
      int * num_calls_ptr = NULL;
      const char * nfname = NULL;
      if (sparseLayer) {
         nfname = "numframes_sparse";
         num_calls_ptr = &writeActivitySparseCalls;
      }
      else {
         nfname = "numframes";
         num_calls_ptr = &writeActivityCalls;
      }
      parent->readScalarFromFile(cpDir, getName(), nfname, num_calls_ptr, 0);
   }
   //Need to exchange border information since lastUpdateTime is being read from checkpoint, so no guarentee that publish will call exchange
   status = icComm->exchangeBorders(this->getLayerId(), this->getLayerLoc());
   status |= icComm->wait(this->getLayerId());
   assert(status == PV_SUCCESS);
   //Update sparse indices here
   status = updateAllActiveIndices();

   return PV_SUCCESS;
}

template<class T>
int HyPerLayer::readBufferFile(const char * filename, InterColComm * comm, double * timeptr, T ** buffers, int numbands, bool extended, const PVLayerLoc * loc) {
   PV_Stream * readFile = pvp_open_read_file(filename, comm);
   int rank = comm->commRank();
   assert( (readFile != NULL && rank == 0) || (readFile == NULL && rank != 0) );
   int numParams = NUM_BIN_PARAMS;
   int params[NUM_BIN_PARAMS];
   int status = pvp_read_header(readFile, comm, params, &numParams);
   if (status != PV_SUCCESS) {
      read_header_err(filename, comm, numParams, params);
   }

   for (int band=0; band<numbands; band++) {
      for(int b = 0; b < loc->nbatch; b++){
         T * bufferBatch;
         if(extended){
            bufferBatch = buffers[band] + b * (loc->nx + loc->halo.rt + loc->halo.lt) * (loc->ny + loc->halo.up + loc->halo.dn) * loc->nf; 
         }
         else{
            bufferBatch = buffers[band] + b * loc->nx * loc->ny * loc->nf;
         }

         double filetime = 0.0;
         switch(params[INDEX_FILE_TYPE]) {
         case PVP_FILE_TYPE:
            filetime = timeFromParams(params);
            break;
         case PVP_ACT_FILE_TYPE:
            status = pvp_read_time(readFile, comm, 0/*root process*/, &filetime);
            if (status!=PV_SUCCESS) {
               fprintf(stderr, "HyPerLayer::readBufferFile error reading timestamp in file \"%s\"\n", filename);
               abort();
            }
            if (rank==0) {
               fprintf(stderr,"HyPerLayer::readBufferFile error: filename \"%s\" is a compressed spiking file, but this filetype has not yet been implemented in this case.\n", filename);
            }
            status = PV_FAILURE;
            break;
         case PVP_NONSPIKING_ACT_FILE_TYPE:
            status = pvp_read_time(readFile, comm, 0/*root process*/, &filetime);
            if (status!=PV_SUCCESS) {
               fprintf(stderr, "HyPerLayer::readBufferFile error reading timestamp in file \"%s\"\n", filename);
               abort();
            }
            break;
         case PVP_WGT_FILE_TYPE:
         case PVP_KERNEL_FILE_TYPE:
            if (rank==0) {
               fprintf(stderr,"HyPerLayer::readBufferFile error: filename \"%s\" is a weight file (type %d) but a layer file is expected.\n", filename, params[INDEX_FILE_TYPE]);
            }
            status = PV_FAILURE;
            break;
         default:
            if (rank==0) {
               fprintf(stderr,"HyPerLayer::readBufferFile error: filename \"%s\" has unrecognized pvp file type %d\n", filename, params[INDEX_FILE_TYPE]);
            }
            status = PV_FAILURE;
            break;
         }
         if (params[INDEX_NX_PROCS] != 1 || params[INDEX_NY_PROCS] != 1) {
            if (rank==0) {
               fprintf(stderr, "HyPerLayer::readBufferFile error: file \"%s\" appears to be in an obsolete version of the .pvp format.\n", filename);
            }
            abort();
         }
         if (status==PV_SUCCESS) {
            status = scatterActivity(readFile, comm, 0/*root process*/, bufferBatch, loc, extended);
         }
         assert(status==PV_SUCCESS);
         if (rank==0 && timeptr && *timeptr != filetime) {
            fprintf(stderr, "Warning: \"%s\" checkpoint has timestamp %g instead of the expected value %g.\n", filename, filetime, *timeptr);
         }
      }
   }
   pvp_close_file(readFile, comm);
   readFile = NULL;
   return status;
}
// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
// template void HyPerCol::ioParamValueRequired<pvdata_t>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, pvdata_t * value);
template int HyPerLayer::readBufferFile<float>(const char * filename, InterColComm * comm, double * timeptr, float ** buffers, int numbands, bool extended, const PVLayerLoc * loc);

int HyPerLayer::readDataStoreFromFile(const char * filename, InterColComm * comm, double * timeptr) {
   PV_Stream * readFile = pvp_open_read_file(filename, comm);
   assert( (readFile != NULL && comm->commRank() == 0) || (readFile == NULL && comm->commRank() != 0) );
   int numParams = NUM_BIN_PARAMS;
   int params[NUM_BIN_PARAMS];
   int status = pvp_read_header(readFile, comm, params, &numParams);
   if (status != PV_SUCCESS) {
      read_header_err(filename, comm, numParams, params);
   }
   if (params[INDEX_NX_PROCS] != 1 || params[INDEX_NY_PROCS] != 1) {
      if (comm->commRank()==0) {
         fprintf(stderr, "HyPerLayer::readBufferFile error: file \"%s\" appears to be in an obsolete version of the .pvp format.\n", filename);
      }
      abort();
   }
   int numlevels = comm->publisherStore(getLayerId())->numberOfLevels();
   int numbuffers = comm->publisherStore(getLayerId())->numberOfBuffers();
   if (params[INDEX_NBANDS] != numlevels*numbuffers) {
      fprintf(stderr, "readDataStoreFromFile error reading \"%s\": number of delays + batches in file is %d, but number of delays + batches in layer is %d\n", filename, params[INDEX_NBANDS], numlevels*numbuffers);
      abort();
   }
   DataStore * datastore = comm->publisherStore(getLayerId());
   for (int b = 0; b < numbuffers; b++){
      for (int l=0; l<numlevels; l++) {
         double tlevel;
         pvp_read_time(readFile, comm, 0/*root process*/, &tlevel);
         datastore->setLastUpdateTime(b/*bufferId*/, l, tlevel);
         pvdata_t * buffer = (pvdata_t *) datastore->buffer(b, l);
         int status1 = scatterActivity(readFile, comm, 0/*root process*/, buffer, getLayerLoc(), true, NULL, 0, 0, PVP_NONSPIKING_ACT_FILE_TYPE, 0);
         if (status1 != PV_SUCCESS) status = PV_FAILURE;
      }
   }
   assert(status == PV_SUCCESS);
   pvp_close_file(readFile, comm);
   return status;
}

int HyPerLayer::checkpointWrite(const char * cpDir) {
   // Writes checkpoint files for V, A, and datastore to files in working directory
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();

   char * filename = NULL;
   filename = parent->pathInCheckpoint(cpDir, getName(), "_A.pvp");
   pvdata_t * A = getActivity();
   writeBufferFile(filename, icComm, timed, &A, 1, /*extended*/true, getLayerLoc());
   if( getV() != NULL ) {
      free(filename);
      filename = parent->pathInCheckpoint(cpDir, getName(), "_V.pvp");
      pvdata_t * V = getV();
      writeBufferFile(filename, icComm, timed, &V, /*numbands*/1, /*extended*/false, getLayerLoc());
   }
   free(filename);
   filename = parent->pathInCheckpoint(cpDir, getName(), "_Delays.pvp");
   writeDataStoreToFile(filename, icComm, timed);
   free(filename);

   parent->writeScalarToFile(cpDir, getName(), "lastUpdateTime", lastUpdateTime);
   parent->writeScalarToFile(cpDir, getName(), "nextUpdateTime", nextUpdateTime);
   parent->writeScalarToFile(cpDir, getName(), "nextWrite", writeTime);

   if (parent->columnId()==0) {
      if (outputStateStream) {
         long activityfilepos = getPV_StreamFilepos(outputStateStream);
         parent->writeScalarToFile(cpDir, getName(), "filepos", activityfilepos);
      }
   }

   if (writeStep>=0.0f) {
      if (sparseLayer) {
         parent->writeScalarToFile(cpDir, getName(), "numframes_sparse", writeActivitySparseCalls);
      }
      else {
         parent->writeScalarToFile(cpDir, getName(), "numframes", writeActivityCalls);
      }
   }

   return PV_SUCCESS;
}

template <typename T>
int HyPerLayer::writeBufferFile(const char * filename, InterColComm * comm, double timed, T ** buffers, int numbands, bool extended, const PVLayerLoc * loc) {
   PV_Stream * writeFile = pvp_open_write_file(filename, comm, /*append*/false);
   assert( (writeFile != NULL && comm->commRank() == 0) || (writeFile == NULL && comm->commRank() != 0) );

   //nbands gets multiplied by loc->nbatches in this function
   int * params = pvp_set_nonspiking_act_params(comm, timed, loc, PV_FLOAT_TYPE, numbands);
   assert(params && params[1]==NUM_BIN_PARAMS);
   int status = pvp_write_header(writeFile, comm, params, NUM_BIN_PARAMS);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "HyPerLayer::writeBufferFile error writing \"%s\"\n", filename);
      abort();
   }

   for (int band=0; band<numbands; band++) {
      for(int b = 0; b < loc->nbatch; b++){
         if (writeFile != NULL) { // Root process has writeFile set to non-null; other processes to NULL.
            int numwritten = PV_fwrite(&timed, sizeof(double), 1, writeFile);
            if (numwritten != 1) {
               fprintf(stderr, "HyPerLayer::writeBufferFile error writing timestamp to \"%s\"\n", filename);
               abort();
            }
         }
         T * bufferBatch;
         if(extended){
            bufferBatch = buffers[band] + b * (loc->nx + loc->halo.rt + loc->halo.lt) * (loc->ny + loc->halo.up + loc->halo.dn) * loc->nf; 
         }
         else{
            bufferBatch = buffers[band] + b * loc->nx * loc->ny * loc->nf;
         }

         status = gatherActivity(writeFile, comm, 0, bufferBatch, loc, extended);
      }
   }
   free(params);
   pvp_close_file(writeFile, comm);
   writeFile = NULL;
   return status;
}
// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
// template void HyPerCol::ioParamValueRequired<pvdata_t>(enum ParamsIOFlag ioFlag, const char * group_name, const char * param_name, pvdata_t * value);
template int HyPerLayer::writeBufferFile<float>(const char * filename, InterColComm * comm, double timed, float ** buffers, int numbands, bool extended, const PVLayerLoc * loc);

int HyPerLayer::writeDataStoreToFile(const char * filename, InterColComm * comm, double timed) {
   PV_Stream * writeFile = pvp_open_write_file(filename, comm, /*append*/false);
   assert( (writeFile != NULL && comm->commRank() == 0) || (writeFile == NULL && comm->commRank() != 0) );
   int numlevels = comm->publisherStore(getLayerId())->numberOfLevels();
   int numbuffers = comm->publisherStore(getLayerId())->numberOfBuffers();
   assert(numlevels == getNumDelayLevels());
   int * params = pvp_set_nonspiking_act_params(comm, timed, getLayerLoc(), PV_FLOAT_TYPE, numlevels);
   assert(params && params[1]==NUM_BIN_PARAMS);
   int status = pvp_write_header(writeFile, comm, params, NUM_BIN_PARAMS);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "HyPerLayer::writeBufferFile error writing \"%s\"\n", filename);
      abort();
   }
   free(params);
   DataStore * datastore = comm->publisherStore(getLayerId());
   for(int b = 0; b < numbuffers; b++){
      for (int l=0; l<numlevels; l++) {
         if (writeFile != NULL) { // Root process has writeFile set to non-null; other processes to NULL.
            double lastUpdateTime = datastore->getLastUpdateTime(b/*bufferId*/, l);
            int numwritten = PV_fwrite(&lastUpdateTime, sizeof(double), 1, writeFile);
            if (numwritten != 1) {
               fprintf(stderr, "HyPerLayer::writeBufferFile error writing timestamp to \"%s\"\n", filename);
               abort();
            }
         }
         pvdata_t * buffer = (pvdata_t *) datastore->buffer(b, l);
         int status1 = gatherActivity(writeFile, comm, 0, buffer, getLayerLoc(), true/*extended*/);
         if (status1 != PV_SUCCESS) status = PV_FAILURE;
      }
   }
   assert(status == PV_SUCCESS);
   pvp_close_file(writeFile, comm);
   writeFile = NULL;
   return status;
}

int HyPerLayer::writeTimers(FILE* stream){
   if (parent->icCommunicator()->commRank()==0) {
      recvsyn_timer->fprint_time(stream);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
      gpu_recvsyn_timer->fprint_time(stream);
#endif
      update_timer->fprint_time(stream);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
      gpu_update_timer->fprint_time(stream);
#endif
      publish_timer->fprint_time(stream);
      timescale_timer->fprint_time(stream);
      io_timer->fprint_time(stream);
      for (int p=0; p<getNumProbes(); p++){
         getProbe(p)->writeTimer(stream);
      }
   }
   return PV_SUCCESS;
}

int HyPerLayer::writeActivitySparse(double timed, bool includeValues)
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   int status = PV::writeActivitySparse(outputStateStream, parent->icCommunicator(), timed, store, getLayerLoc(), includeValues);

   if (status == PV_SUCCESS) {
      status = incrementNBands(&writeActivitySparseCalls);
   }
   return status;
}

// write non-spiking activity
int HyPerLayer::writeActivity(double timed)
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   int status = PV::writeActivity(outputStateStream, parent->icCommunicator(), timed, store, getLayerLoc());

   if (status == PV_SUCCESS) {
      status = incrementNBands(&writeActivityCalls);
   }
   return status;
}

int HyPerLayer::incrementNBands(int * numCalls) {
   // Only the root process needs to maintain INDEX_NBANDS, so only the root process modifies numCalls
   // This way, writeActivityCalls does not need to be coordinated across MPI
   int status;
   if( parent->icCommunicator()->commRank() == 0 ) {
      assert(outputStateStream!=NULL);
      (*numCalls) = (*numCalls) + parent->getNBatch();
      long int fpos = getPV_StreamFilepos(outputStateStream);
      PV_fseek(outputStateStream, sizeof(int)*INDEX_NBANDS, SEEK_SET);
      int intswritten = PV_fwrite(numCalls, sizeof(int), 1, outputStateStream);
      PV_fseek(outputStateStream, fpos, SEEK_SET);
      status = intswritten == 1 ? PV_SUCCESS : PV_FAILURE;
   }
   else {
      status = PV_SUCCESS;
   }
   return status;
}


bool HyPerLayer::localDimensionsEqual(PVLayerLoc const * loc1, PVLayerLoc const * loc2) {
   return
         loc1->nbatch==loc2->nbatch &&
         loc1->nx==loc2->nx &&
         loc1->ny==loc2->ny &&
         loc1->nf==loc2->nf &&
         loc1->halo.lt==loc2->halo.lt &&
         loc1->halo.rt==loc2->halo.rt &&
         loc1->halo.dn==loc2->halo.dn &&
         loc1->halo.up==loc2->halo.up;
}

int HyPerLayer::mirrorToNorthWest(PVLayerCube * dest, PVLayerCube * src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int nbatch = dest->loc.nbatch;
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder = dest->loc.halo.up;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;

      pvdata_t * src0 = srcData+ topBorder*sy + leftBorder*sx;
      pvdata_t * dst0 = srcData+ (topBorder - 1)*sy + (leftBorder - 1)*sx;

      for (int ky = 0; ky < topBorder; ky++) {
         pvdata_t * to   = dst0 - ky*sy;
         pvdata_t * from = src0 + ky*sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToNorth(PVLayerCube * dest, PVLayerCube * src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int nx = clayer->loc.nx;
   int nf = clayer->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder = dest->loc.halo.up;
   int nbatch = dest->loc.nbatch;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;
      pvdata_t * src0 = srcData+ topBorder*sy + leftBorder*sx;
      pvdata_t * dst0 = destData+ (topBorder-1)*sy + leftBorder*sx;

      for (int ky = 0; ky < topBorder; ky++) {
         pvdata_t * to   = dst0 - ky*sy;
         pvdata_t * from = src0 + ky*sy;
         for (int kx = 0; kx < nx; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to += nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToNorthEast(PVLayerCube* dest, PVLayerCube* src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int nx = dest->loc.nx;
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int rightBorder = dest->loc.halo.rt;
   int topBorder = dest->loc.halo.up;
   int nbatch = dest->loc.nbatch;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;
      pvdata_t * src0 = srcData + topBorder*sy + (nx + leftBorder - 1)*sx;
      pvdata_t * dst0 = destData + (topBorder-1)*sy + (nx + leftBorder)*sx;

      for (int ky = 0; ky < topBorder; ky++) {
         pvdata_t * to   = dst0 - ky*sy;
         pvdata_t * from = src0 + ky*sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToWest(PVLayerCube* dest, PVLayerCube* src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int ny = dest->loc.ny;
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder = dest->loc.halo.up;
   int nbatch = dest->loc.nbatch;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;
      pvdata_t * src0 = srcData + topBorder*sy + leftBorder*sx;
      pvdata_t * dst0 = destData + topBorder*sy + (leftBorder - 1)*sx;

      for (int ky = 0; ky < ny; ky++) {
         pvdata_t * to   = dst0 + ky*sy;
         pvdata_t * from = src0 + ky*sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToEast(PVLayerCube* dest, PVLayerCube* src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int nx = clayer->loc.nx;
   int ny = clayer->loc.ny;
   int nf = clayer->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int rightBorder = dest->loc.halo.rt;
   int topBorder = dest->loc.halo.up;
   int nbatch = dest->loc.nbatch;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;
      pvdata_t * src0 = srcData + topBorder*sy + (nx + leftBorder - 1)*sx;
      pvdata_t * dst0 = destData + topBorder*sy + (nx + leftBorder)*sx;

      for (int ky = 0; ky < ny; ky++) {
         pvdata_t * to   = dst0 + ky*sy;
         pvdata_t * from = src0 + ky*sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToSouthWest(PVLayerCube* dest, PVLayerCube* src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int ny = dest->loc.ny;
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder = dest->loc.halo.up;
   int bottomBorder = dest->loc.halo.dn;
   int nbatch = dest->loc.nbatch;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;
      pvdata_t * src0 = srcData + (ny + topBorder - 1)*sy + leftBorder*sx;
      pvdata_t * dst0 = destData + (ny + topBorder)*sy + (leftBorder - 1)*sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         pvdata_t * to   = dst0 + ky*sy;
         pvdata_t * from = src0 - ky*sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToSouth(PVLayerCube* dest, PVLayerCube* src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int rightBorder = dest->loc.halo.rt;
   int topBorder = dest->loc.halo.up;
   int bottomBorder = dest->loc.halo.dn;
   int nbatch = dest->loc.nbatch;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;
      pvdata_t * src0 = srcData + (ny + topBorder -1)*sy + leftBorder*sx;
      pvdata_t * dst0 = destData + (ny + topBorder)*sy + leftBorder*sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         pvdata_t * to   = dst0 + ky*sy;
         pvdata_t * from = src0 - ky*sy;
         for (int kx = 0; kx < nx; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to += nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToSouthEast(PVLayerCube* dest, PVLayerCube* src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int rightBorder = dest->loc.halo.rt;
   int topBorder = dest->loc.halo.up;
   int bottomBorder = dest->loc.halo.dn;
   int nbatch = dest->loc.nbatch;
   size_t sb = strideBExtended(&dest->loc);
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   for(int b=0; b<nbatch; b++){
      pvdata_t* srcData = src->data + b*sb ;
      pvdata_t* destData = dest->data + b*sb;
      pvdata_t * src0 = srcData + (ny + topBorder - 1)*sy + (nx + leftBorder - 1)*sx;
      pvdata_t * dst0 = destData + (ny + topBorder)*sy + (nx + leftBorder)*sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         pvdata_t * to   = dst0 + ky*sy;
         pvdata_t * from = src0 - ky*sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf*sf] = from[kf*sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
   return 0;
}


//Removed functions

//int HyPerLayer::getNumMargin(){
//   if (marginIndices == NULL){
//      getMarginIndices();
//   }
//   return numMargin;
//}
//
//int * HyPerLayer::getMarginIndices(){
//   if (marginIndices == NULL){
//      int kMargin = 0;
//      const PVLayerLoc * layerLoc = getLayerLoc();
//      const int marginUp = layerLoc->halo.up;
//      const int marginDn = layerLoc->halo.dn;
//      const int marginLt = layerLoc->halo.lt;
//      const int marginRt = layerLoc->halo.rt;
//      numMargin = marginUp * marginDn * marginLt * marginRt;
//      assert(numMargin == getNumExtended() - getNumNeurons());
//      const int nf = layerLoc->nf;
//      const int nx = layerLoc->nx;
//      const int ny = layerLoc->ny;
//      int nxExt = nx + marginRt + marginLt;
//      int nyExt = ny + marginUp + marginDn;
//      //int syExt = nf * nxExt;
//      //int sxExt = nf;
//      int * marginIndices = (int *) calloc(numMargin, sizeof(int));
//      assert(marginIndices != NULL);
//      // get North margin indices
//      for (int kPreExt = 0; kPreExt < nf * nxExt * marginUp; kPreExt++) {
//         marginIndices[kMargin++] = kPreExt;
//      }
//      assert(kMargin == nf * nxExt * marginUp);
//      // get East margin indices
//      for (int ky = marginUp; ky < marginUp + ny; ky++) {
//         for (int kx = 0; kx < marginLt; kx++) {
//            for (int kf = 0; kf < nf; kf++) {
//               int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
//               marginIndices[kMargin++] = kPreExt;
//            }
//         }
//      }
//      assert(kMargin == nf * nxExt * marginUp + nf * marginLt * ny);
//      // get West margin indices
//      for (int ky = marginUp; ky < marginUp + ny; ky++) {
//         for (int kx = nx + marginLt; kx < nxExt; kx++) {
//            for (int kf = 0; kf < nf; kf++) {
//               int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
//               marginIndices[kMargin++] = kPreExt;
//            }
//         }
//      }
//      assert(kMargin == nf * nxExt * marginUp + nf * marginLt * ny + nf * marginUp * ny);
//      // get South margin indices
//      for (int kPreExt = kMargin; kPreExt < numMargin; kPreExt++) {
//         marginIndices[kMargin++] = kPreExt;
//      }
//      assert(kMargin == numMargin);
//   }
//   return marginIndices;
//}

//// copyDirect is never called.  Do we still need it?
///* copy src PVLayerCube to dest PVLayerCube */
///* initialize src, dest to beginning of data structures */
//int copyDirect(pvdata_t * dest, pvdata_t * src, int nf, int nxSrc, int nySrc, int syDst, int sySrc)
//{
//   pvdata_t * to   = dest;
//   pvdata_t * from = src;
//
//   for (int j = 0; j < nySrc; j++) {
//      to   = dest + j*syDst;
//      from = src  + j*sySrc;
//      for (int i = 0; i < nxSrc; i++) {
//         for (int f = 0; f < nf; f++) {
//            to[f] = from[f];
//         }
//         to   += nf;
//         from += nf;
//      }
//   }
//   return 0;
//}
//// Template functions
////
//template <typename T>
//int HyPerLayer::copyFromBuffer(const T * buf, T * data,
//      const PVLayerLoc * loc, bool extended, T scale)
//{
//   size_t sf, sx, sy;
//
//   const int nx = loc->nx;
//   const int ny = loc->ny;
//   const int nf = loc->nf;
//
//   int nxBorder = 0;
//   int nyBorder = 0;
//
//   if (extended) {
//      nxBorder = loc->halo.lt;
//      nyBorder = loc->halo.up;
//      sf = strideFExtended(loc);
//      sx = strideXExtended(loc);
//      sy = strideYExtended(loc);
//   }
//   else {
//      sf = strideF(loc);
//      sx = strideX(loc);
//      sy = strideY(loc);
//   }
//
//   int ii = 0;
//   for (int j = 0; j < ny; j++) {
//      int jex = j + nyBorder;
//      for (int i = 0; i < nx; i++) {
//         int iex = i + nxBorder;
//         for (int f = 0; f < nf; f++) {
//            data[iex*sx + jex*sy + f*sf] = scale * buf[ii++];
//         }
//      }
//   }
//   return 0;
//}

//#ifdef PV_USE_OPENCL
//size_t HyPerLayer::getLayerDataStoreOffset(int delay)
//{
//   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
//   size_t offset  = store->bufferOffset(LOCAL, delay);
//   // (Rasmussen) still sorting this out
//   // size_t offset2 = (store->bufferOffset(0, 0) - store->bufferOffset(LOCAL, delay));
//   return offset;
//}
//
//int HyPerLayer::copyDataStoreCLBuffer() {
//   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
//   return store->copyBufferToDevice();
//}
//int HyPerLayer::waitForDataStoreCopy() {
//   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
//   return store->waitForCopy();
//}
//
//CLBuffer * HyPerLayer::getLayerDataStoreCLBuffer()
//{
//   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
//   return store->getCLBuffer();
//}
//
////int HyPerLayer::initializeDataStoreThreadBuffers()
////{
////   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
////   int status= store->initializeThreadBuffers(parent);
////   //status |= store->getCLBuffer()->copyToDevice(evCopyDataStore);
////   return status;
////}
//
//#endif


//// deprecated 6/16
///**
// * returns the number of neurons in the layer or border region
// * @param borderId the id of the border region (0 for interior/self)
// **/
//int HyPerLayer::numberOfNeurons(int borderId)
//{
//   int numNeurons;
//   const int nx = clayer->loc.nx;
//   const int ny = clayer->loc.ny;
//   const int nf = clayer->loc.nf;
//   const PVHalo * halo = &clayer->loc.halo;
//
//   switch (borderId) {
//   case 0:
//      numNeurons = clayer->numNeurons;         break;
//   case NORTHWEST:
//      numNeurons = halo->lt * halo->up * nf;   break;
//   case NORTH:
//      numNeurons = nx       * halo->up * nf;   break;
//   case NORTHEAST:
//      numNeurons = halo->rt * halo->up * nf;   break;
//   case WEST:
//      numNeurons = halo->lt * ny       * nf;   break;
//   case EAST:
//      numNeurons = halo->rt * ny       * nf;   break;
//   case SOUTHWEST:
//      numNeurons = halo->lt * halo->dn * nf;   break;
//   case SOUTH:
//      numNeurons = nx       * halo->dn * nf;   break;
//   case SOUTHEAST:
//      numNeurons = halo->rt * halo->dn * nf;   break;
//   default:
//      fprintf(stderr, "ERROR:HyPerLayer:numberOfBorderNeurons: bad border index %d\n", borderId);
//      numNeurons = 0; break;
//   }
//
//   return numNeurons;
//}

//Deprecated 6/16/15
//int HyPerLayer::gatherToInteriorBuffer(unsigned char * buf)
//{
//   return HyPerLayer::copyToBuffer(buf, getLayerData(), getLayerLoc(), isExtended(), 255.0);
//}
//
//int HyPerLayer::copyToBuffer(unsigned char * buf, const pvdata_t * data,
//      const PVLayerLoc * loc, bool extended, float scale)
//{
//   size_t sb, sf, sx, sy;
//
//   const int nx = loc->nx;
//   const int ny = loc->ny;
//   const int nf = loc->nf;
//   const int nbatch = loc->nbatch
//
//   int leftBorder = 0;
//   int topBorder = 0;
//
//   if (extended) {
//      leftBorder = loc->halo.lt;
//      topBorder = loc->halo.up;
//      sf = strideFExtended(loc);
//      sx = strideXExtended(loc);
//      sy = strideYExtended(loc);
//      sb = strideBExtended(loc);
//   }
//   else {
//      sf = strideF(loc);
//      sx = strideX(loc);
//      sy = strideY(loc);
//      sb = strideB(loc);
//   }
//
//   int ii = 0;
//   for (int j = 0; j < ny; j++) {
//      int jex = j + topBorder;
//      for (int i = 0; i < nx; i++) {
//         int iex = i + leftBorder;
//         for (int f = 0; f < nf; f++) {
//            buf[ii++] = (unsigned char) (scale * data[iex*sx + jex*sy + f*sf]);
//         }
//      }
//   }
//   return 0;
//}
//
//int HyPerLayer::copyToBuffer(pvdata_t * buf, const pvdata_t * data,
//      const PVLayerLoc * loc, bool extended, float scale)
//{
//   size_t sf, sx, sy;
//   int leftBorder, topBorder;
//
//   const int nx = loc->nx;
//   const int ny = loc->ny;
//   const int nf = loc->nf;
//
//   if (extended) {
//      leftBorder = loc->halo.lt;
//      topBorder = loc->halo.up;
//      sf = strideFExtended(loc);
//      sx = strideXExtended(loc);
//      sy = strideYExtended(loc);
//   }
//   else {
//      leftBorder = 0;
//      topBorder = 0;
//      sf = strideF(loc);
//      sx = strideX(loc);
//      sy = strideY(loc);
//   }
//
//   int ii = 0;
//   for (int j = 0; j < ny; j++) {
//      int jex = j + topBorder;
//      for (int i = 0; i < nx; i++) {
//         int iex = i + leftBorder;
//         for (int f = 0; f < nf; f++) {
//            buf[ii++] = scale * data[iex*sx + jex*sy + f*sf];
//         }
//      }
//   }
//   return 0;
//}
//
//int HyPerLayer::copyFromBuffer(const unsigned char * buf, pvdata_t * data,
//      const PVLayerLoc * loc, bool extended, float scale)
//{
//   size_t sf, sx, sy;
//
//   const int nx = loc->nx;
//   const int ny = loc->ny;
//   const int nf = loc->nf;
//
//   int leftBorder = 0;
//   int topBorder = 0;
//
//   if (extended) {
//      leftBorder = loc->halo.lt;
//      topBorder = loc->halo.up;
//      sf = strideFExtended(loc);
//      sx = strideXExtended(loc);
//      sy = strideYExtended(loc);
//   }
//   else {
//      sf = strideF(loc);
//      sx = strideX(loc);
//      sy = strideY(loc);
//   }
//
//   int ii = 0;
//   for (int j = 0; j < ny; j++) {
//      int jex = j + topBorder;
//      for (int i = 0; i < nx; i++) {
//         int iex = i + leftBorder;
//         for (int f = 0; f < nf; f++) {
//            data[iex*sx + jex*sy + f*sf] = scale * (pvdata_t) buf[ii++];
//         }
//      }
//   }
//   return 0;
//}

//Labels deprecated 6/16/15
///**
// * Return the label (if any) of a neuron in this layer.  A label may be the
// * orientation (for example) of a neuron.  Creating a label for a neuron is
// * normally done by offline analysis after the synaptic weights for connections
// * to the layer have been learned.
// */
//int HyPerLayer::label(int k)
//{
//   if (labels == NULL) return 0;
//   else                return labels[k];
//}
//
//int HyPerLayer::calcActiveIndices() {
//   //Active indicies stored as local ext values
//   int numActive = 0;
//   PVLayerLoc & loc = clayer->loc;
//   pvdata_t * activity = clayer->activity->data;
//
//   for (int kex = 0; kex < getNumExtended(); kex++) {
//      if (activity[kex] != 0.0) {
//         clayer->activeIndices[numActive++] = kex;
//      }
//   }
//   //for (int k = 0; k < getNumNeurons(); k++) {
//   //   const int kex = kIndexExtended(k, loc.nx, loc.ny, loc.nf, loc.halo.lt, loc.halo.rt, loc.halo.dn, loc.halo.up);
//   //   if (activity[kex] != 0.0) {
//   //      clayer->activeIndices[numActive++] = globalIndexFromLocal(k, loc);
//   //   }
//   //}
//   clayer->numActive = numActive;
//
//   return PV_SUCCESS;
//}

//int HyPerLayer::updateV() {
//   pvdata_t * V = getV();
//   pvdata_t * GSynExc = getChannel(CHANNEL_EXC);
//   pvdata_t * GSynInh = getChannel(CHANNEL_INH);
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      V[k] = GSynExc[k] - GSynInh[k];
//   }
//   return PV_SUCCESS;
//}

//int HyPerLayer::allocateActiveIndices() {
//   //Active indicies is local ext
//   return allocateBuffer(&clayer->activeIndices, getNumExtended(), "active indices");
//}




} // end of PV namespace

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifndef PV_USE_OPENCL
#  include "../kernels/HyPerLayer_recv_post.cl"
#endif


#ifdef __cplusplus
}
#endif // __cplusplus


