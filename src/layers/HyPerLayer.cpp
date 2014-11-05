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
   this->labels = NULL;
   this->marginIndices = NULL;
   this->numMargin = 0;
   this->writeTime = 0;
   this->initialWriteTime = 0;
   this->triggerFlag = false; //Default to update every timestamp
   this->triggerLayer = NULL;
   this->triggerLayerName = NULL;
   this->initVObject = NULL;
   this->triggerOffset = 0;
   this->nextUpdateTime = 0;
   this->initializeFromCheckpointFlag = false;
   this->restartFlag = false; // Deprecated July 31, 2014 in favor of initializeFromCheckpointFlag
   
   this->lastUpdateTime = 0.0;
   this->phase = 0;

   this->initInfoCommunicatedFlag = false;
   this->dataStructuresAllocatedFlag = false;
   this->initialValuesSetFlag = false;
   
   this->numSynchronizedMarginWidthLayers = 0;
   this->synchronizedMarginWidthLayers = NULL;

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
   this->name = strdup(name);
   setParent(hc); // Could this line and the parent->addLayer line be combined in a HyPerLayer method?

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

   int status = ioParams(PARAMS_IO_READ);
   assert(status == PV_SUCCESS);

   writeTime = initialWriteTime;
   writeActivityCalls = 0;
   writeActivitySparseCalls = 0;
   numDelayLevels = 1; // If a connection has positive delay so that more delay levels are needed, numDelayLevels is increased when HyPerConn::communicateInitInfo calls increaseDelayLevels
   maxRate = 1000.0f/parent->getDeltaTime();

   initClayer();

   // must set ioAppend before addLayer is called (addLayer causes activity file to be opened using layerid)
   ioAppend = parent->getCheckpointReadFlag() ? 1 : 0;

   layerId = parent->addLayer(this);

   status = openOutputStateFile();

   lastUpdateTime = parent->simulationTime();
   nextUpdateTime = parent->getDeltaTime();

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
   setLayerLoc(loc, nxScale, nyScale, numFeatures);
   assert(loc->halo.lt==0 && loc->halo.rt==0 && loc->halo.dn==0 && loc->halo.up==0);

   clayer->numNeurons  = loc->nx * loc->ny * loc->nf;
   clayer->numExtended = clayer->numNeurons; // initially, margin is zero; it will be updated as needed during the communicateInitInfo stage.

   double xScaled = -log2( (double) nxScale);
   double yScaled = -log2( (double) nyScale);

   int xScale = (int) nearbyint(xScaled);
   int yScale = (int) nearbyint(yScaled);

   clayer->xScale = xScale;
   clayer->yScale = yScale;

   clayer->dx = powf(2.0f, (float) xScale);
   clayer->dy = powf(2.0f, (float) yScale);

   clayer->xOrigin = 0.5 + clayer->loc.kx0 * clayer->dx;
   clayer->yOrigin = 0.5 + clayer->loc.ky0 * clayer->dy;

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

   delete initVObject; initVObject = NULL;
   freeClayer();
   free(name); name = NULL;
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

   free(labels); labels = NULL;
   free(marginIndices); marginIndices = NULL;
   for (int i_probe = 0; i_probe < this->numProbes; i_probe++){
      delete probes[i_probe];
   }
   free(probes);

   free(synchronizedMarginWidthLayers);
   if(triggerLayerName){
      free(triggerLayerName);
      triggerLayerName = NULL;
   }

   if(thread_gSyn){
      for(int i = 0; i < parent->getNumThreads(); i++){
         free(thread_gSyn[i]);
         thread_gSyn[i] = NULL;
      }
      free(thread_gSyn);
      thread_gSyn = NULL;
   }

}

int HyPerLayer::freeClayer() {
   pvcube_delete(clayer->activity);

   if (clayer->activeFP != NULL) {
      PV_fclose(clayer->activeFP);
      clayer->activeFP = NULL;
   }

   if (clayer->posFP != NULL) {
      PV_fclose(clayer->posFP);
      clayer->posFP = NULL;
   }

   //free(clayer->activeIndices); clayer->activeIndices = NULL;
   free(clayer->prevActivity);  clayer->prevActivity = NULL;
   //free(clayer->activeIndices); clayer->activeIndices = NULL;
   free(clayer->V);             clayer->V = NULL;
   free(clayer);                clayer = NULL;

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
   for (k = 0; k < getNumExtended(); k++) {
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

int HyPerLayer::allocateV() {
   return allocateBuffer(&clayer->V, getNumNeurons(), "membrane potential V");
}

int HyPerLayer::allocateActivity() {
   clayer->activity = pvcube_new(&clayer->loc, getNumExtended());
   return clayer->activity!=NULL ? PV_SUCCESS : PV_FAILURE;
}

//int HyPerLayer::allocateActiveIndices() {
//   //Active indicies is local ext
//   return allocateBuffer(&clayer->activeIndices, getNumExtended(), "active indices");
//}

int HyPerLayer::allocatePrevActivity() {
   return allocateBuffer(&clayer->prevActivity, getNumExtended(), "time of previous activity");
}

int HyPerLayer::setLayerLoc(PVLayerLoc * layerLoc, float nxScale, float nyScale, int nf)
{
   int status = PV_SUCCESS;

   InterColComm * icComm = parent->icCommunicator();

   float nxglobalfloat = nearbyintf(nxScale * parent->getNxGlobal());
   layerLoc->nxGlobal = (int) nxglobalfloat;
   if (fabs(nxglobalfloat-layerLoc->nxGlobal)>0.0001) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Size of column is compatible with nxScale of layer \"%s\".\n", getName());
         fprintf(stderr, "Column nx %d multiplied by nxScale %f must be an integer.\n", parent->getNxGlobal(), nxScale);
      }
      status = PV_FAILURE;
   }

   float nyglobalfloat = nearbyintf(nyScale * parent->getNyGlobal());
   layerLoc->nyGlobal = (int) nyglobalfloat;
   if (fabs(nxglobalfloat-layerLoc->nxGlobal)>0.0001) {
      if (parent->columnId()==0) {
         fprintf(stderr, "Size of column is compatible with nyScale of layer \"%s\".\n", getName());
         fprintf(stderr, "Column ny %d multiplied by nyScale %f must be an integer.\n", parent->getNyGlobal(), nxScale);
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
#ifdef PV_USE_MPI
   MPI_Barrier(icComm->communicator()); // If there is an error, make sure that MPI doesn't kill the run before process 0 reports the error.
#endif
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

      GSyn[0] = (pvdata_t *) calloc(getNumNeurons()*numChannels, sizeof(pvdata_t));
      // All channels allocated at once and contiguously.  resetGSynBuffers_HyPerLayer() assumes this is true, to make it easier to port to GPU.
      if(GSyn[0] == NULL) {
         status = PV_FAILURE;
         return status;
      }

      for (int m = 1; m < numChannels; m++) {
         GSyn[m] = GSyn[0] + m * getNumNeurons();
      }
   }

   return status;
}

int HyPerLayer::initializeState() {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();

   assert(!params->presentAndNotBeenRead(name, "initializeFromCheckpointFlag"));
   if (initializeFromCheckpointFlag) {
      assert(parent->getInitializeFromCheckpointDir() && parent->getInitializeFromCheckpointDir()[0]);
      status = readStateFromCheckpoint(parent->getInitializeFromCheckpointDir(), NULL);
   }
   else {
      assert(!params->presentAndNotBeenRead(name, "restart"));
      if( restartFlag ) {
         status = readState(NULL);
         if(!parent->getCheckpointReadFlag()){
            nextUpdateTime = parent->getDeltaTime();
            //updateNextUpdateTime();
         }
      }
      else {
         status = setInitialValues();
      }
   }
   return status;
}

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
      //Moved to publish
      //if (status == PV_SUCCESS) status = updateActiveIndices();
   }
   return status;
}

int HyPerLayer::initializeActivity() {
   int status = setActivity();
   //Moved to publish
   //if (status == PV_SUCCESS) {
   //   status = updateActiveIndices();
   //}
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
   ioParam_marginWidth(ioFlag);
   ioParam_phase(ioFlag);
   ioParam_mirrorBCflag(ioFlag);
   ioParam_valueBC(ioFlag);
   ioParam_initializeFromCheckpointFlag(ioFlag);
   ioParam_restart(ioFlag); // Deprecated July 31, 2014 in favor of initializeFromCheckpointFlag
   ioParam_InitVType(ioFlag);
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_sparseLayer(ioFlag);
   ioParam_writeSparseValues(ioFlag);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   ioParam_updateGpu(ioFlag);
#endif // PV_USE_OPENCL
   return PV_SUCCESS;
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
void HyPerLayer::ioParam_updateGpu(enum ParamsIOFlag ioFlag){
   parent->ioParamValue(ioFlag, name, "updateGpu", &updateGpu, updateGpu);
#ifdef PV_USE_OPENCL
   if(updateGpu){
      std::cout << "Updating from gpu is not implemented with opencl yet\n";
   }
#endif
}
#endif

void HyPerLayer::ioParam_nxScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nxScale", &nxScale, nxScale);
}

void HyPerLayer::ioParam_nyScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nyScale", &nyScale, nyScale);
}

void HyPerLayer::ioParam_nf(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nf", &numFeatures, numFeatures);
}

void HyPerLayer::ioParam_marginWidth(enum ParamsIOFlag ioFlag) {
   // marginWidth parameter was deprecated July 25, 2013.
   // As of Aug 12, 2014, marginWidth parameter is no longer read.
   // After enough time has passed, this function should be deleted.
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(name, "marginWidth")) {
      if (parent->columnId()==0) {
         fprintf(stderr, "HyPerLayer \"%s\": margins are adjusted automatically; parameter marginWidth is no longer read.\n", name);
      }
   }
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

void HyPerLayer::ioParam_restart(enum ParamsIOFlag ioFlag) {
   if (parent->parameters()->present(name, "restart")) {
      parent->ioParamValue(ioFlag, name, "restart", &restartFlag, false/*default value*/);
      // restart was deprecated July 31, 2014
      if (parent->parameters()->present(name, "restart") && parent->columnId()==0) {
         fprintf(stderr, " *** %s \"%s\": parameter \"restart\" has been deprecated.\n", parent->parameters()->groupKeywordFromName(getName()), getName());
         if (restartFlag) {
            fprintf(stderr, " ***     Instead of restart=true, set HyPerCol's initializeFromCheckpointDir to the output/Last directory,\n");
            fprintf(stderr, " ***     and set each layer's initializeFromCheckpointFlag according to whether or not to load that layer from the checkpoint.\n");
         }
      }
   }
}

void HyPerLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initVObject = new InitV(parent, name);
      if( initVObject == NULL ) {
         fprintf(stderr, "%s \"%s\" error: unable to create InitV object\n", parent->parameters()->groupKeywordFromName(name), name);
         abort();
      }
   }
   if (initVObject != NULL) {
      initVObject->ioParamsFillGroup(ioFlag);
   }
}

void HyPerLayer::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "triggerFlag", &triggerFlag, triggerFlag);
}

void HyPerLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerFlag"));
   if (triggerFlag) {
      parent->ioParamStringRequired(ioFlag, name, "triggerLayerName", &triggerLayerName);
   }
}

void HyPerLayer::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerFlag"));
   if (triggerFlag) {
      parent->ioParamValue(ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
      if(triggerOffset < 0){
         fprintf(stderr, "%s \"%s\" error in rank %d process: TriggerOffset (%f) must be positive\n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId(), triggerOffset);
         exit(EXIT_FAILURE);
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
                  parent->parameters()->groupKeywordFromName(name), name);
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
      parent->ioParamValue(ioFlag, name, "writeSparseValues", &writeSparseValues, false/*default value*/);
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

int HyPerLayer::allocateUpdateKernel(){
   std::cout << "Layer " << name << " of type " << parent->parameters()->groupKeywordFromName(getName()) << " does not support updating on gpus yet\n"; 
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

   
   const size_t size    = getNumNeurons()  * sizeof(float);
   const size_t size_ex = getNumExtended() * sizeof(float);

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
      d_ActiveIndices = device->createBuffer(CL_MEM_READ_ONLY, size_ex, NULL);
#endif
#ifdef PV_USE_CUDA
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
   // HyPerCol also calls each HyPerConn's communicateInitInfo() method, which
   // (among other things) calls its presynaptic layer's requireMarginWidth().
   // Since all communicateInitInfo() methods are called before any allocateDataStructures()
   // methods, HyPerLayer knows its marginWidth before it has to allocate
   // anything.  So it no longer needs to be specified in params!
   if(triggerFlag){
      triggerLayer = parent->getLayerFromName(triggerLayerName);
      if (triggerLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
                  parent->parameters()->groupKeywordFromName(name), name, triggerLayerName);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
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

int HyPerLayer::openOutputStateFile() {
   char filename[PV_PATH_MAX];
   char posFilename[PV_PATH_MAX];
   switch( parent->includeLayerName() ) {
   case 0:
      snprintf(filename, PV_PATH_MAX, "%s/a%d.pvp", parent->getOutputPath(), layerId);
      break;
   case 1:
      snprintf(filename, PV_PATH_MAX, "%s/a%d_%s.pvp", parent->getOutputPath(), layerId, name);
      break;
   case 2:
      snprintf(filename, PV_PATH_MAX, "%s/%s.pvp", parent->getOutputPath(), name);
      break;
   default:
      assert(0);
      break;
   }

   if(sparseLayer){
      switch( parent->includeLayerName() ) {
      case 0:
         snprintf(posFilename, PV_PATH_MAX, "%s/a%d.pos", parent->getOutputPath(), layerId);
         break;
      case 1:
         snprintf(posFilename, PV_PATH_MAX, "%s/a%d_%s.pos", parent->getOutputPath(), layerId, name);
         break;
      case 2:
         snprintf(posFilename, PV_PATH_MAX, "%s/%s.pos", parent->getOutputPath(), name);
         break;
      default:
         assert(0);
         break;
      }
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
#ifdef PV_USE_MPI
   MPI_Bcast(&ioAppend, 1, MPI_INT, 0/*root*/, icComm->communicator());
#endif
   clayer->activeFP = pvp_open_write_file(filename, icComm, ioAppend);
   if(sparseLayer){
      clayer->posFP = pvp_open_write_file(posFilename, icComm, ioAppend);
   }
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

   //Doing this check here, since trigger layers are being set up in communicate init info
   //If the magnitude of the trigger offset is bigger than the delta update time, then error
   if(triggerFlag){
      double deltaUpdateTime = getDeltaUpdateTime();
      if(deltaUpdateTime != -1 && triggerOffset >= deltaUpdateTime){ 
         fprintf(stderr, "%s \"%s\" error in rank %d process: TriggerOffset (%f) must be lower than the change in update time (%f) \n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId(), triggerOffset, deltaUpdateTime);
         exit(EXIT_FAILURE);
      }
   }
   //updateNextUpdateTime();

   allocateClayerBuffers();

   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   PVHalo const * halo = &loc->halo;

   // If not mirroring, fill the boundaries with the value in the valueBC param
   if (!useMirrorBCs() && getValueBC()!=0.0f) {
      int idx = 0;
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
      assert(idx==getNumExtended());
   }

   // allocate storage for the input conductance arrays
   status = allocateBuffers();
   assert(status == PV_SUCCESS);

   // labels are not extended
   labels = (int *) calloc(getNumNeurons(), sizeof(int));
   if (labels==NULL) {
      fprintf(stderr, "HyPerLayer \"%s\" error: rank %d unable to allocate memory for labels.\n", name, parent->columnId());
      exit(EXIT_FAILURE);
   }

   //Allocate temp buffers if needed, 1 for each thread
   if(parent->getNumThreads() > 1){
      thread_gSyn = (pvdata_t**) malloc(sizeof(pvdata_t*) * parent->getNumThreads());
      assert(thread_gSyn);

      //Assign thread_gSyn to different points of tempMem
      for(int i = 0; i < parent->getNumThreads(); i++){
         pvdata_t* tempMem = (pvdata_t*) malloc(sizeof(pvdata_t) * getNumNeurons());
         if(!tempMem){
            fprintf(stderr, "HyPerLayer \"%s\" error: rank %d unable to allocate %zu memory for thread_gSyn: %s\n", name, parent->columnId(), sizeof(pvdata_t) * getNumNeurons(), strerror(errno));
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

   // do allocation stage for probes
   for (int i=0; i<numProbes; i++) {
      LayerProbe * p = probes[i];
      if (p==NULL) continue;
      int pstatus = p->allocateDataStructures();
      if (pstatus==PV_SUCCESS) {
         if (parent->columnId()==0) printf("Probe \"%s\" allocateDataStructures completed.\n", p->getName());
      }
      else {
         assert(pstatus == PV_FAILURE); // PV_POSTPONE etc. hasn't been implemented for probes yet.
         exit(EXIT_FAILURE); // Any error message should be printed by probe's communicateInitInfo function
      }
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

// getLastUpdateTime() method for base class.
// Default behavior is to update every timestep.  For layers that update less frequently, they should
// save parent->simulationTime() to lastUpdateTime whenever they do update, and override getLastUpdateTime
// to return lastUpdateTime without setting it to parent->simulationTime().
//
// Publisher calls getLastUpdateTime and compares it to simulationTime() before doing border exchanges over MPI, so managing
// lastUpdateTime can save a lot of MPI traffic.
//
// One wrinkle is that all layers call updateState before any layers call publish, so lastUpdateTime could be one timestep behind
// if you depend on getLastUpdateTime to set lastUpdateTime.  If this is an issue, lastUpdateTime should be set in updateState.
double HyPerLayer::getLastUpdateTime() {
   //Taken out, now handled in updateStateWrapper
   //lastUpdateTime=parent->simulationTime();
   return lastUpdateTime;
}

/**
 * Returns the activity data for the layer.  This data is in the
 * extended space (with margins).
 */
const pvdata_t * HyPerLayer::getLayerData(int delay)
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   return (pvdata_t *) store->buffer(LOCAL, delay);
}

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


// deprecated?
/**
 * returns the number of neurons in the layer or border region
 * @param borderId the id of the border region (0 for interior/self)
 **/
int HyPerLayer::numberOfNeurons(int borderId)
{
   int numNeurons;
   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const PVHalo * halo = &clayer->loc.halo;

   switch (borderId) {
   case 0:
      numNeurons = clayer->numNeurons;         break;
   case NORTHWEST:
      numNeurons = halo->lt * halo->up * nf;   break;
   case NORTH:
      numNeurons = nx       * halo->up * nf;   break;
   case NORTHEAST:
      numNeurons = halo->rt * halo->up * nf;   break;
   case WEST:
      numNeurons = halo->lt * ny       * nf;   break;
   case EAST:
      numNeurons = halo->rt * ny       * nf;   break;
   case SOUTHWEST:
      numNeurons = halo->lt * halo->dn * nf;   break;
   case SOUTH:
      numNeurons = nx       * halo->dn * nf;   break;
   case SOUTHEAST:
      numNeurons = halo->rt * halo->dn * nf;   break;
   default:
      fprintf(stderr, "ERROR:HyPerLayer:numberOfBorderNeurons: bad border index %d\n", borderId);
      numNeurons = 0; break;
   }

   return numNeurons;
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

int HyPerLayer::gatherToInteriorBuffer(unsigned char * buf)
{
   return HyPerLayer::copyToBuffer(buf, getLayerData(), getLayerLoc(), isExtended(), 255.0);
}

int HyPerLayer::copyToBuffer(unsigned char * buf, const pvdata_t * data,
      const PVLayerLoc * loc, bool extended, float scale)
{
   size_t sf, sx, sy;

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   int leftBorder = 0;
   int topBorder = 0;

   if (extended) {
      leftBorder = loc->halo.lt;
      topBorder = loc->halo.up;
      sf = strideFExtended(loc);
      sx = strideXExtended(loc);
      sy = strideYExtended(loc);
   }
   else {
      sf = strideF(loc);
      sx = strideX(loc);
      sy = strideY(loc);
   }

   int ii = 0;
   for (int j = 0; j < ny; j++) {
      int jex = j + topBorder;
      for (int i = 0; i < nx; i++) {
         int iex = i + leftBorder;
         for (int f = 0; f < nf; f++) {
            buf[ii++] = (unsigned char) (scale * data[iex*sx + jex*sy + f*sf]);
         }
      }
   }
   return 0;
}

int HyPerLayer::copyToBuffer(pvdata_t * buf, const pvdata_t * data,
      const PVLayerLoc * loc, bool extended, float scale)
{
   size_t sf, sx, sy;
   int leftBorder, topBorder;

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   if (extended) {
      leftBorder = loc->halo.lt;
      topBorder = loc->halo.up;
      sf = strideFExtended(loc);
      sx = strideXExtended(loc);
      sy = strideYExtended(loc);
   }
   else {
      leftBorder = 0;
      topBorder = 0;
      sf = strideF(loc);
      sx = strideX(loc);
      sy = strideY(loc);
   }

   int ii = 0;
   for (int j = 0; j < ny; j++) {
      int jex = j + topBorder;
      for (int i = 0; i < nx; i++) {
         int iex = i + leftBorder;
         for (int f = 0; f < nf; f++) {
            buf[ii++] = scale * data[iex*sx + jex*sy + f*sf];
         }
      }
   }
   return 0;
}

int HyPerLayer::copyFromBuffer(const unsigned char * buf, pvdata_t * data,
      const PVLayerLoc * loc, bool extended, float scale)
{
   size_t sf, sx, sy;

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   int leftBorder = 0;
   int topBorder = 0;

   if (extended) {
      leftBorder = loc->halo.lt;
      topBorder = loc->halo.up;
      sf = strideFExtended(loc);
      sx = strideXExtended(loc);
      sy = strideYExtended(loc);
   }
   else {
      sf = strideF(loc);
      sx = strideX(loc);
      sy = strideY(loc);
   }

   int ii = 0;
   for (int j = 0; j < ny; j++) {
      int jex = j + topBorder;
      for (int i = 0; i < nx; i++) {
         int iex = i + leftBorder;
         for (int f = 0; f < nf; f++) {
            data[iex*sx + jex*sy + f*sf] = scale * (pvdata_t) buf[ii++];
         }
      }
   }
   return 0;
}


bool HyPerLayer::needUpdate(double time, double dt){
   //Always update on first timestep
   //if (time <= parent->getStartTime()){
   //    return true;
   //}

   //This function needs to return true if the layer was updated this timestep as well
   if(fabs(parent->simulationTime() - lastUpdateTime) < (dt/2)){
      return true;
   }
   //Never update flag
   //If nextUpdateTime is -1, the layer won't update
   if(nextUpdateTime == -1){
      return false;
   }
   //Check based on nextUpdateTime and triggerOffset
   //Needs to be a equality check, so to account for roundoff errors, we check if it's within half the delta time
   if(fabs(time - (nextUpdateTime - triggerOffset)) < (dt/2)){
      return true;
   }
   return false;



   ////If layer is a trigger flag, call the attached trigger layer's needUpdate
   //if(triggerFlag){
   //   assert(triggerLayer);
   //   if (getPhase() > triggerLayer->getPhase()) {
   //      return triggerLayer->getLastUpdateTime() >= lastUpdateTime;
   //   }
   //   else {
   //      return triggerLayer->getLastUpdateTime() > lastUpdateTime;
   //   }
   //}
   ////Otherwise, needs to update every timestep
   //else{
   //   return true;
   //}
}

int HyPerLayer::updateNextUpdateTime(){
   double deltaUpdateTime = getDeltaUpdateTime();
   assert(deltaUpdateTime != 0);
   if(deltaUpdateTime != -1){
      while(parent->simulationTime() >= nextUpdateTime){
         nextUpdateTime += deltaUpdateTime;
      }
   }
   else{
      //Never update
      nextUpdateTime = -1;
   }
   return PV_SUCCESS;
}

double HyPerLayer::getDeltaUpdateTime(){
   if(triggerFlag){
      assert(triggerLayer);
      return triggerLayer->getDeltaUpdateTime();
   }
   else{
      return parent->getDeltaTime();
   }
}

int HyPerLayer::updateStateWrapper(double timef, double dt){
   int status = PV_SUCCESS;
   //   if(needUpdate(timef, dt)){
   if(needUpdate(timef, parent->getDeltaTime())){
#ifdef PV_USE_OPENCL
      //If this current layer's gsyn is on the gpu, only move it back when doing update state or output state
      this->clFinishGSyn();
#endif
      update_timer->start();
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
      if(updateGpu){
         gpu_update_timer->start();
         status = updateStateGpu(timef, dt);
         gpu_update_timer->stop();
      }
      else{
#endif
         status = updateState(timef, dt);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
      }
      //Activity updated, set flag to true
      updatedDeviceActivity = true;
      updatedDeviceDatastore = true;
#endif
      lastUpdateTime=parent->simulationTime();
      update_timer->stop();
   }
   //Because of the triggerOffset, we need to check if we need to update nextUpdateTime every time
   updateNextUpdateTime();
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

   //Moved to publish
   //if(status == PV_SUCCESS) status = updateActiveIndices();
   return status;
}


int HyPerLayer::resetGSynBuffers(double timef, double dt) {
   int status = PV_SUCCESS;
   if (GSyn == NULL) return PV_SUCCESS;
   resetGSynBuffers_HyPerLayer(this->getNumNeurons(), getNumChannels(), GSyn[0]); // resetGSynBuffers();
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
   int num_neurons = nx*ny*nf;
   if (num_channels == 1){
      applyGSyn_HyPerLayer1Channel(num_neurons, V, gSynHead);
   }
   else{
      applyGSyn_HyPerLayer(num_neurons, V, gSynHead);
   }
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);

   return PV_SUCCESS;
}

int HyPerLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_HyPerLayer(getNumNeurons(), clayer->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
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

//int HyPerLayer::updateV() {
//   pvdata_t * V = getV();
//   pvdata_t * GSynExc = getChannel(CHANNEL_EXC);
//   pvdata_t * GSynInh = getChannel(CHANNEL_INH);
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      V[k] = GSynExc[k] - GSynInh[k];
//   }
//   return PV_SUCCESS;
//}


//This function must be called after exchange borders and a wait
int HyPerLayer::updateActiveIndices() {
   return parent->icCommunicator()->updateActiveIndices(this->getLayerId());
}

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

float HyPerLayer::getConvertToRateDeltaTimeFactor(HyPerConn* conn)
{
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   float dt_factor = 1.0f;
   bool preActivityIsNotRate = conn->preSynapticActivityIsNotRate();
   if (preActivityIsNotRate) {
      enum ChannelType channel_type = conn->getChannel();
      float dt = getParent()->getDeltaTime();
      float tau = this->getChannelTimeConst(channel_type);
      if (tau > 0) {
         double exp_dt_tau = exp(-dt / tau);
         dt_factor = (1 - exp_dt_tau) / exp_dt_tau;
         // the above factor ensures that for a constant input of G_SYN to an excitatory conductance G_EXC,
         // then G_EXC -> G_SYN as t -> inf
      }
      else {
         dt_factor = dt;
      }
   }
   return dt_factor;
}

int HyPerLayer::recvAllSynapticInput() {
   int status = PV_SUCCESS;
   //Only recvAllSynapticInput if we need an update
   if(needUpdate(parent->simulationTime(), parent->getDeltaTime())){
      //int numConnections = parent->numberOfConnections();
      //for (int c=0; c<numConnections; c++) {
         //HyPerConn * conn = parent->getConnection(c);
         //if (conn->postSynapticLayer()!=this) continue;
      bool switchGpu = false;
      //Start CPU timer here
      recvsyn_timer->start();

      for(std::vector<HyPerConn*>::iterator it = recvConns.begin(); it < recvConns.end(); it++){
         HyPerConn * conn = *it;
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

         //Check if updating from post perspective
         HyPerLayer * pre = conn->preSynapticLayer();
         PVLayerCube cube;
         memcpy(&cube.loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
         cube.numItems = pre->getNumExtended();
         cube.size = sizeof(PVLayerCube);

         DataStore * store = parent->icCommunicator()->publisherStore(pre->getLayerId());
         int numArbors = conn->numberOfAxonalArborLists();

         for (int arbor=0; arbor<numArbors; arbor++) {
            int delay = conn->getDelay(arbor);
            cube.data = (pvdata_t *) store->buffer(LOCAL, delay);
            if(!conn->getUpdateGSynFromPostPerspective()){
               cube.isSparse = store->isSparse();
               if(cube.isSparse){
                  cube.numActive = *(store->numActiveBuffer(LOCAL, delay));
                  cube.activeIndices = store->activeIndicesBuffer(LOCAL, delay);
               }
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
               if(conn->getReceiveGpu()){
                  status = recvSynapticInputGpu(conn, &cube, arbor);
                  //No need to update GSyn since it's already living on gpu
                  updatedDeviceGSyn = false;
               }
               else
#endif
               {
                  status = recvSynapticInput(conn, &cube, arbor);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
                  //CPU updated gsyn, need to update gsyn
                  updatedDeviceGSyn = true;
#endif
               }
            }
            else{
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
               if(conn->getReceiveGpu()){
                  status = recvSynapticInputFromPostGpu(conn, &cube, arbor);
                  updatedDeviceGSyn = false;
               }
               else
#endif
               {
                  status = recvSynapticInputFromPost(conn, &cube, arbor);
#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
                  updatedDeviceGSyn = true;
#endif
               }
            }
            assert(status == PV_SUCCESS || status == PV_BREAK);
            if (status == PV_BREAK){
               break;
            }
         }
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
float HyPerLayer::syncGpu(){
   float time = 0;

   if(recvGpu || updateGpu){
#ifdef PV_USE_CUDA
      parent->getDevice()->syncDevice();
#endif
#ifdef PV_USE_OPENCL
      parent->getDevice()->syncDevice();
#endif
   }

   if(recvGpu){
      time += gpu_recvsyn_timer->accumulateTime();
   }
   if(updateGpu){
      time += gpu_update_timer->accumulateTime();
      
      //Moved to publish
      //int status = updateActiveIndices();
      //assert(status == PV_SUCCESS);
   }
   return time;
}
#endif

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
void HyPerLayer::copyAllGSynToDevice(){
   if(recvGpu || updateGpu){
      //Copy it to device
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




/**
 * Get synaptic input from pre synaptic layer by looping over post synaptic neurons
 * Source layer is pre layer in current connection, post layer in original connection
 * Target layer is post layer in current connection, pre layer in original connection
 * Current layer is target layer
 * cube is activity buffer of source layer
 * conn is the connection from source to target
 */
int HyPerLayer::recvSynapticInputFromPost(HyPerConn * conn, const PVLayerCube * activity, int arborID)
{

   //Check channel number for noupdate
   if(conn->getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(GSyn && GSyn[conn->getChannel()]);

   //Cast to transpose conn
   TransposeConn * sourceToTargetConn = dynamic_cast <TransposeConn*> (conn);
   if(sourceToTargetConn == NULL){
      fprintf(stderr, "HyPerLayer \"%s\": Updating GSyn buffer from post perspective requires connection %s to be a TransposeConn.\n", name, conn->getName());
      abort();
   }
   //update conn to original connection
   HyPerConn * targetToSourceConn = sourceToTargetConn->getOriginalConn();
   // Don't need TransposeConn to have the same pre and post as originalConn but flipped.  nx,ny,nf must be consistent, but that's checked in initialization.
    ////Assert that the transpose is opposite of the original connection
    //if(targetToSourceConn->preSynapticLayer()->getLayerId() != sourceToTargetConn->postSynapticLayer()->getLayerId() ||
    //   targetToSourceConn->postSynapticLayer()->getLayerId() != sourceToTargetConn->preSynapticLayer()->getLayerId()){
    //   fprintf(stderr, "HyPerLayer \"%s\": Transpose connection %s must be the same connection in the opposite direction of %s.\n", name, sourceToTargetConn->getName(), conn->getName());
    //   abort();
    //}

   assert(arborID >= 0);
   //Get number of neurons restricted target
   const int numRestricted = getNumNeurons();

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::pullSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numRestricted, activity, this, sourceToTargetConn);
   fflush(stdout);
#endif // DEBUG_OUTPUT

   float dt_factor = getConvertToRateDeltaTimeFactor(sourceToTargetConn);

   const PVLayerLoc * oSourceLoc = targetToSourceConn->postSynapticLayer()->getLayerLoc();
   const PVLayerLoc * oTargetLoc = targetToSourceConn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * aSourceLoc = sourceToTargetConn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * aTargetLoc = getLayerLoc();

   const int sourceNx = aSourceLoc->nx;
   const int sourceNy = aSourceLoc->ny;
   const int sourceNf = aSourceLoc->nf;
   const int targetNx = aTargetLoc->nx;
   const int targetNy = aTargetLoc->ny;
   const int targetNf = aTargetLoc->nf;

   const PVHalo * aSourceHalo = &aSourceLoc->halo;
   const PVHalo * oSourceHalo = &oSourceLoc->halo;
   const PVHalo * aTargetHalo = &aTargetLoc->halo;
   const PVHalo * oTargetHalo = &oTargetLoc->halo;

   //get source layer's extended y stride
   int sy  = (sourceNx+aSourceHalo->lt+aSourceHalo->rt)*sourceNf;
   //get source layer's patch y stride
   int syp = targetToSourceConn->yPatchStride(); // Should be correct even if targetToSourceConn points to a different layer than sourceToTargetConn's pre.
   //Iterate through y patch
   int numPerStride = targetToSourceConn->xPatchSize() * targetToSourceConn->fPatchSize();

   //The start of the gsyn buffer
   pvdata_t * gSynPatchHead = this->getChannel(sourceToTargetConn->getChannel());

   long * startSourceExtBuf = conn->getPostToPreActivity();
   if(!startSourceExtBuf){
      std::cout << "HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?\n";
      exit(EXIT_FAILURE);
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int kTargetRes = 0; kTargetRes < numRestricted; kTargetRes++){
      //Change restricted to extended post neuron
      int akTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, aTargetHalo->lt, aTargetHalo->rt, aTargetHalo->dn, aTargetHalo->up);
      int okTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, oTargetHalo->lt, oTargetHalo->rt, oTargetHalo->dn, oTargetHalo->up);

      bool inWindow; 
      inWindow = inWindowExt(arborID, akTargetExt);
      if(!inWindow) continue;

      //Read from buffer
      long startSourceExt = startSourceExtBuf[kTargetRes];

      //Calculate target's start of gsyn
      pvdata_t * gSynPatchPos = gSynPatchHead + kTargetRes;

      int kernelIndex = targetToSourceConn->patchToDataLUT(okTargetExt);
      uint4 * rngPtr = conn->getRandState(kTargetRes);

      for (int ky = 0; ky < targetToSourceConn->yPatchSize(); ky++){
         float * activityY = &(activity->data[startSourceExt + ky*sy]);
         pvwdata_t * weightY = targetToSourceConn->get_wDataHead(arborID, kernelIndex) + ky*syp;
         (conn->accumulateFunctionFromPostPointer)(numPerStride, gSynPatchPos, activityY, weightY, dt_factor, rngPtr);
      }
   }
   return PV_SUCCESS;
}

/**
 * Receive synaptic input from pre synaptic layer by looping over pre synaptic neurons 
 */
int HyPerLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID)
{
   //Check if we need to update based on connection's channel
   if(conn->getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(GSyn && GSyn[conn->getChannel()]);

   float dt_factor = getConvertToRateDeltaTimeFactor(conn);

   const PVLayerLoc * preLoc = conn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = this->getLayerLoc();


   assert(arborID >= 0);
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
   fflush(stdout);
#endif // DEBUG_OUTPUT


   //Clear all thread gsyn buffer
   if(thread_gSyn){
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int i = 0; i < parent->getNumThreads() * getNumNeurons(); i++){
         int ti = i/getNumNeurons();
         int ni = i % getNumNeurons();
         thread_gSyn[ti][ni] = 0;
      }
   }

   int numLoop;
   if(activity->isSparse){
      numLoop = activity->numActive;
   }
   else{
      numLoop = numExtended;
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
   //TODO loop over active indicies here instead
   for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
      int kPre;
      if(activity->isSparse){
         kPre = activity->activeIndices[loopIndex];
      }
      else{
         kPre = loopIndex;
      }

      bool inWindow; 
      //Post layer recieves synaptic input
      //Only with respect to post layer
      const PVLayerLoc * preLoc = conn->preSynapticLayer()->getLayerLoc();
      const PVLayerLoc * postLoc = this->getLayerLoc();
      int kPost = layerIndexExt(kPre, preLoc, postLoc);
      inWindow = inWindowExt(arborID, kPost);
      if(!inWindow) continue;

      float a = activity->data[kPre] * dt_factor;
      // Activity < 0 is used by generative models --pete
      if (a == 0.0f) continue;

      //If we're using thread_gSyn, set this here
      pvdata_t * gSynPatchHead;
#ifdef PV_USE_OPENMP_THREADS
      if(thread_gSyn){
         int ti = omp_get_thread_num();
         gSynPatchHead = thread_gSyn[ti];
      }
      else{
         gSynPatchHead = this->getChannel(conn->getChannel());
      }
#else
      gSynPatchHead = this->getChannel(conn->getChannel());
#endif
      recvOnePreNeuronActivity(conn, kPre, arborID, a, gSynPatchHead, conn->getRandState(kPre));
   }
#ifdef PV_USE_OPENMP_THREADS
   //Accumulate back into gSyn
   if(thread_gSyn){
      pvdata_t * gSynPatchHead = this->getChannel(conn->getChannel());
      //Looping over neurons first to be thread safe
#pragma omp parallel for
      for(int ni = 0; ni < getNumNeurons(); ni++){
         //Different for maxpooling
         if(conn->getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
            for(int ti = 0; ti < parent->getNumThreads(); ti++){
               gSynPatchHead[ni] = gSynPatchHead[ni] < thread_gSyn[ti][ni] ? thread_gSyn[ti][ni] : gSynPatchHead[ni];
            }
         }
         else{
            for(int ti = 0; ti < parent->getNumThreads(); ti++){
               gSynPatchHead[ni] += thread_gSyn[ti][ni];
            }
         }
      }
   }
#endif

   return PV_SUCCESS;
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)

int HyPerLayer::recvSynapticInputFromPostGpu(HyPerConn * conn, const PVLayerCube * activity, int arborID)
{
   //Check channel number for noupdate
   if(conn->getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(GSyn && GSyn[conn->getChannel()]);

   //Cast to transpose conn
   TransposeConn * sourceToTargetConn = dynamic_cast <TransposeConn*> (conn);
   if(sourceToTargetConn == NULL){
      fprintf(stderr, "HyPerLayer \"%s\": Updating GSyn buffer from post perspective requires connection %s to be a TransposeConn.\n", name, conn->getName());
      abort();
   }
   //update conn to original connection
   HyPerConn * targetToSourceConn = sourceToTargetConn->getOriginalConn();

   assert(arborID >= 0);
   //Get number of neurons restricted target
   const int numRestricted = getNumNeurons();

   float dt_factor = getConvertToRateDeltaTimeFactor(sourceToTargetConn);

   const PVLayerLoc * oSourceLoc = targetToSourceConn->postSynapticLayer()->getLayerLoc();
   const PVLayerLoc * oTargetLoc = targetToSourceConn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * aSourceLoc = sourceToTargetConn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * aTargetLoc = getLayerLoc();
   const PVHalo * aSourceHalo = &aSourceLoc->halo;

   const int sourceNx = aSourceLoc->nx;
   const int sourceNy = aSourceLoc->ny;
   const int sourceNf = aSourceLoc->nf;
   const int targetNx = aTargetLoc->nx;
   const int targetNy = aTargetLoc->ny;
   const int targetNf = aTargetLoc->nf;

   //get source layer's extended y stride
   int sy  = (sourceNx+aSourceHalo->rt+aSourceHalo->lt)*sourceNf;
   //get source layer's patch y stride
   int syp = targetToSourceConn->yPatchStride(); // Should be correct even if targetToSourceConn points to a different layer than sourceToTargetConn's pre.
   //Iterate through y patch
   int numPerStride = targetToSourceConn->xPatchSize() * targetToSourceConn->fPatchSize();

   long * startSourceExtBuf = conn->getPostToPreActivity();
   if(!startSourceExtBuf){
      std::cout << "HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?\n";
      exit(EXIT_FAILURE);
   }

#ifdef PV_USE_OPENCL
   //Grab kernel from conn
   CLKernel * krRecvPost = conn->getKrRecvPost();        // CL kernel for update state call
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaRecvPost * krRecvPost = conn->getKrRecvPost();        // CL kernel for update state call
#endif
   assert(krRecvPost);

   bool updatePreAct = false;
   //Update pre activity, post gsyn, and conn weights 
   //Only if their updated
   if(sourceToTargetConn->preSynapticLayer()->getUpdatedDeviceDatastoreFlag()){
      float * h_preDatastore = activity->data;
#ifdef PV_USE_OPENCL
      CLBuffer * d_preDatastore = sourceToTargetConn->preSynapticLayer()->getDeviceDatastore();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer* d_preDatastore = sourceToTargetConn->preSynapticLayer()->getDeviceDatastore();
#endif
      assert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);
      //Device now has updated 
      sourceToTargetConn->preSynapticLayer()->setUpdatedDeviceDatastoreFlag(false);
      updatePreAct = true;
   }
   

   bool updateWeights = false;
   if(targetToSourceConn->getUpdatedDeviceWFlag()){
      //These weights should be orig conn weights
      float * h_weights = targetToSourceConn->get_wDataStart(arborID);

#ifdef PV_USE_OPENCL
      CLBuffer * d_weights = targetToSourceConn->getDeviceWData();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_weights = targetToSourceConn->getDeviceWData();
#endif
      assert(d_weights);
      d_weights->copyToDevice(h_weights);
      targetToSourceConn->setUpdatedDeviceWFlag(false);
      updateWeights = true;
   }

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   if(updatePreAct){
      krRecvPost->permuteDatastorePVToCudnn();
   }
   if(updateWeights){
      krRecvPost->permuteWeightsPVToCudnn();
   }
   //Permute GSyn 
   krRecvPost->permuteGSynPVToCudnn(sourceToTargetConn->getChannel());
#endif

   int totF = targetNf;
   int totX = targetNx;
   int totY = targetNy;
   //Make sure local sizes are divisible by f, x, and y
   //krRecvPost->run(numRestricted, 0, NULL, NULL);
#ifdef PV_USE_OPENCL
   if(conn->getNumFLocal() != 1){
      printf("gpu post run: numFLocal must be 1\n");
      exit(-1);
   }
   if(conn->getNumYLocal() != 1){
      printf("gpu post run: numYLocal must be 1\n");
      exit(-1);
   }
   cl_event* timerEvent;
   timerEvent = this->gpu_recvsyn_timer->getStartEvent();
   krRecvPost->run(totF, totX, totY, conn->getNumFLocal(), conn->getNumXLocal(), conn->getNumYLocal(),
         0, NULL, timerEvent);
#endif
#ifdef PV_USE_CUDA
   krRecvPost->run(totX, totY, totF, conn->getNumXLocal(), conn->getNumYLocal(), conn->getNumFLocal());
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   krRecvPost->permuteGSynCudnnToPV(sourceToTargetConn->getChannel());
#endif

   return PV_SUCCESS;
}

int HyPerLayer::recvSynapticInputGpu(HyPerConn * conn, const PVLayerCube * activity, int arborID){
   //Check if we need to update based on connection's channel
   if(conn->getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(GSyn && GSyn[conn->getChannel()]);

   float dt_factor = getConvertToRateDeltaTimeFactor(conn);

   //Post layer recieves synaptic input
   //Only with respect to post layer
   const PVLayerLoc * preLoc = conn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = this->getLayerLoc();
   //If the connection uses gpu to receive, update all buffers

   //TODO see if you can avoid this step of transfering patches to gpu
   //Based on arborId
   //Other way would be to just allocate all arbors to gpu
   
   //If more than 1 arbor, need to update patches and GSynPatchStart.
   //If one arbor, done in allocatePreKernel in HyPerConn
   if(conn->numberOfAxonalArborLists() > 1){
      PVPatch* h_patches = conn->weights(arborID)[0]; //0 beacuse it's one block of memory
#ifdef PV_USE_OPENCL
      CLBuffer * d_patches = conn->getDevicePatches();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_patches = conn->getDevicePatches();
#endif
      assert(d_patches);

      d_patches->copyToDevice(h_patches);

      size_t* h_GSynPatchStart = conn->getGSynPatchStart()[arborID];
#ifdef PV_USE_OPENCL
      CLBuffer * d_GSynPatchStart = conn->getDeviceGSynPatchStart();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_GSynPatchStart = conn->getDeviceGSynPatchStart();
#endif
      assert(d_GSynPatchStart);
      d_GSynPatchStart->copyToDevice(h_GSynPatchStart);
   }

   //Update pre datastore, post gsyn, and conn weights 
   //Only if their updated
   if(conn->preSynapticLayer()->getUpdatedDeviceDatastoreFlag()){
      float * h_preDatastore= activity->data;
#ifdef PV_USE_OPENCL
      CLBuffer * d_preDatastore= conn->preSynapticLayer()->getDeviceDatastore();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_preDatastore = conn->preSynapticLayer()->getDeviceDatastore();
#endif
      assert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);

      //Copy active indices and num active if needed
      if(activity->isSparse){
#ifdef PV_USE_OPENCL
         CLBuffer * d_ActiveIndices;
#endif
#ifdef PV_USE_CUDA
         PVCuda::CudaBuffer * d_ActiveIndices;
#endif
         d_ActiveIndices = conn->preSynapticLayer()->getDeviceActiveIndices();
         assert(d_ActiveIndices);
         unsigned int * h_ActiveIndices = activity->activeIndices;
         unsigned int h_numActive = activity->numActive;
         assert(h_ActiveIndices);
         d_ActiveIndices->copyToDevice(h_ActiveIndices, h_numActive * sizeof(unsigned int));
      }
      //Device now has updated 
      conn->preSynapticLayer()->setUpdatedDeviceDatastoreFlag(false);
   }
   
   if(conn->getUpdatedDeviceWFlag()){
      float * h_weights = conn->get_wDataStart(arborID);
#ifdef PV_USE_OPENCL
      CLBuffer * d_weights = conn->getDeviceWData();
#endif
#ifdef PV_USE_CUDA
      PVCuda::CudaBuffer * d_weights = conn->getDeviceWData();
#endif
      assert(d_weights);
      d_weights->copyToDevice(h_weights);
      conn->setUpdatedDeviceWFlag(false);
   }
   
#ifdef PV_USE_OPENCL
   //Grab kernel from conn
   CLKernel * krRecvPre = conn->getKrRecvPre();        // CL kernel for update state call
#endif
#ifdef PV_USE_CUDA
   PVCuda::CudaKernel * krRecvPre = conn->getKrRecvPre();        // CL kernel for update state call
#endif
   assert(krRecvPre);
   
   //int totX = conn->getNumPostGroupX();
   //int totY = conn->getNumPostGroupY();
   
   //X direction is active neuron
   //Y direction is post patch size
   long totActiveNeuron;
   if(activity->isSparse){
      totActiveNeuron = activity->numActive;
   }
   else{
      totActiveNeuron = conn->preSynapticLayer()->getNumExtended();
   }

   long totPatchSize = conn->xPatchSize() * conn->yPatchSize() * conn->fPatchSize();

   long totThreads = totActiveNeuron * totPatchSize;

#ifdef PV_USE_OPENCL
   cl_event* timerEvent;
   timerEvent = this->gpu_recvsyn_timer->getStartEvent();
   std::cout << "opencl recv pre not implented yet\n";
   exit(-1);

   //krRecvPre->run(totX, totY, conn->getNumXLocal(), conn->getNumYLocal(),
   //     0, NULL, timerEvent);
#endif

#ifdef PV_USE_CUDA
   int maxThreads = parent->getDevice()->get_max_threads();
   int numLocalThreads = totPatchSize < maxThreads ? totPatchSize : maxThreads;
   
   krRecvPre->run_nocheck(totThreads, numLocalThreads);
#endif

   return PV_SUCCESS;
}


#endif

void HyPerLayer::recvOnePreNeuronActivity(HyPerConn * conn, int patchIndex, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr) {
   PVPatch * weights = conn->getWeights(patchIndex, arbor);
   const int nk = weights->nx * conn->fPatchSize();
   const int ny = weights->ny;
   const int sy  = conn->getPostNonextStrides()->sy;       // stride in layer
   const int syw = conn->yPatchStride();                   // stride in patch
   pvwdata_t * weightDataStart = conn->get_wData(arbor,patchIndex); // make this a pvwdata_t const *?
   pvgsyndata_t * postPatchStart = postBufferStart + conn->getGSynPatchStart(patchIndex, arbor);

   for (int y = 0; y < ny; y++) {
      (conn->accumulateFunctionPointer)(nk, postPatchStart + y*sy, a, weightDataStart + y*syw, auxPtr);
   }
}

int HyPerLayer::publish(InterColComm* comm, double time)
{
   publish_timer->start();

   if ( useMirrorBCs()&& getLastUpdateTime() >= getParent()->simulationTime()) { //needUpdate(parent->simulationTime(), parent->getDeltaTime()) ) { //
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

//
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
      fprintf(stderr, "%s \"%s\": outputState failed on rank %d process.\n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId());
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
   //status = updateActiveIndices();
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
      if (parent->columnId()==0) {
         assert(clayer->activeFP);
         if (PV_fseek(clayer->activeFP, activityfilepos, SEEK_SET) != 0) {
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
   status = updateActiveIndices();

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
      for (int band=0; band<numbands; band++) {
         status = scatterActivity(readFile, comm, 0/*root process*/, buffers[band], loc, extended);
      }
   }
   assert(status==PV_SUCCESS);
   pvp_close_file(readFile, comm);
   readFile = NULL;
   if (rank==0 && timeptr && *timeptr != filetime) {
      fprintf(stderr, "Warning: \"%s\" checkpoint has timestamp %g instead of the expected value %g.\n", filename, filetime, *timeptr);
   }
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
   if (params[INDEX_NBANDS] != numlevels) {
      fprintf(stderr, "readDataStoreFromFile error reading \"%s\": number of delays in file is %d, but number of delays in layer is %d\n", filename, params[INDEX_NBANDS], numlevels);
      abort();
   }
   DataStore * datastore = comm->publisherStore(getLayerId());
   for (int l=0; l<numlevels; l++) {
      double tlevel;
      pvp_read_time(readFile, comm, 0/*root process*/, &tlevel);
      if (comm->commRank()==0 && timeptr != NULL && *timeptr != tlevel) {
         fprintf(stderr, "Warning: \"%s\" delay level %d has timestamp %g instead of the expected value %g.\n", filename, l, tlevel, *timeptr);
      }
      pvdata_t * buffer = (pvdata_t *) datastore->buffer(0, l);
      int status1 = scatterActivity(readFile, comm, 0/*root process*/, buffer, getLayerLoc(), true);
      if (status1 != PV_SUCCESS) status = PV_FAILURE;
   }
   assert(status == PV_SUCCESS);
   pvp_close_file(readFile, comm);
   return status;
}

int HyPerLayer::checkpointWrite(const char * cpDir) {
   // Writes checkpoint files for V, A, and datastore to files in working directory
   InterColComm * icComm = parent->icCommunicator();
   char basepath[PV_PATH_MAX];
   char filename[PV_PATH_MAX];
   int lenbase = snprintf(basepath, PV_PATH_MAX, "%s/%s", cpDir, name);
   if (lenbase+strlen("_Delays.pvp") >= PV_PATH_MAX) { // currently _Delays.pvp is the longest suffix needed
      if (icComm->commRank()==0) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error in layer \"%s\".  Base pathname \"%s/%s_\" too long.\n", name, cpDir, name);
      }
      abort();
   }
   double timed = (double) parent->simulationTime();
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_A.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   pvdata_t * A = getActivity();
   writeBufferFile(filename, icComm, timed, &A, 1, /*extended*/true, getLayerLoc());
   if( getV() != NULL ) {
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s_V.pvp", basepath);
      assert(chars_needed < PV_PATH_MAX);
      pvdata_t * V = getV();
      writeBufferFile(filename, icComm, timed, &V, /*numbands*/1, /*extended*/false, getLayerLoc());
   }
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Delays.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeDataStoreToFile(filename, icComm, timed);

   parent->writeScalarToFile(cpDir, getName(), "lastUpdateTime", lastUpdateTime);
   parent->writeScalarToFile(cpDir, getName(), "nextUpdateTime", nextUpdateTime);
   parent->writeScalarToFile(cpDir, getName(), "nextWrite", writeTime);

   if (parent->columnId()==0) {
      if (clayer->activeFP) {
         long activityfilepos = getPV_StreamFilepos(clayer->activeFP);
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

   int * params = pvp_set_nonspiking_act_params(comm, timed, loc, PV_FLOAT_TYPE, numbands);
   assert(params && params[1]==NUM_BIN_PARAMS);
   int status = pvp_write_header(writeFile, comm, params, NUM_BIN_PARAMS);
   if (status != PV_SUCCESS) {
      fprintf(stderr, "HyPerLayer::writeBufferFile error writing \"%s\"\n", filename);
      abort();
   }
   if (writeFile != NULL) { // Root process has writeFile set to non-null; other processes to NULL.
      int numwritten = PV_fwrite(&timed, sizeof(double), 1, writeFile);
      if (numwritten != 1) {
         fprintf(stderr, "HyPerLayer::writeBufferFile error writing timestamp to \"%s\"\n", filename);
         abort();
      }
   }
   for (int band=0; band<numbands; band++) {
      status = gatherActivity(writeFile, comm, 0, buffers[band], loc, extended);
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
   for (int l=0; l<numlevels; l++) {
      if (writeFile != NULL) { // Root process has writeFile set to non-null; other processes to NULL.
         int numwritten = PV_fwrite(&timed, sizeof(double), 1, writeFile);
         if (numwritten != 1) {
            fprintf(stderr, "HyPerLayer::writeBufferFile error writing timestamp to \"%s\"\n", filename);
            abort();
         }
      }
      pvdata_t * buffer = (pvdata_t *) datastore->buffer(0, l);
      int status1 = gatherActivity(writeFile, comm, 0, buffer, getLayerLoc(), true/*extended*/);
      if (status1 != PV_SUCCESS) status = PV_FAILURE;
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

// Deprecated July 31, 2014 in favor of readStateFromCheckpoint
int HyPerLayer::readState(double * timeptr)
{
   char last_dir[PV_PATH_MAX];
   int chars_needed = snprintf(last_dir, PV_PATH_MAX, "%s/Last", parent->getOutputPath());
   if (chars_needed >= PV_PATH_MAX) {
      if (parent->icCommunicator()->commRank()==0) {
         fprintf(stderr, "HyPerLayer::readState error: path \"%s/Last\" too long.\n", parent->getOutputPath());
      }
      abort();
   }
   return checkpointRead(last_dir, timeptr);
}

int HyPerLayer::writeActivitySparse(double timed, bool includeValues)
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   int status = PV::writeActivitySparse(clayer->activeFP, clayer->posFP, parent->icCommunicator(), timed, store, getLayerLoc(), includeValues);

   if (status == PV_SUCCESS) {
      status = incrementNBands(&writeActivitySparseCalls);
   }
   return status;
}

// write non-spiking activity
int HyPerLayer::writeActivity(double timed)
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   int status = PV::writeActivity(clayer->activeFP, parent->icCommunicator(), timed, store, getLayerLoc());

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
      ++*numCalls;
      long int fpos = getPV_StreamFilepos(clayer->activeFP);
      PV_fseek(clayer->activeFP, sizeof(int)*INDEX_NBANDS, SEEK_SET);
      int intswritten = PV_fwrite(numCalls, sizeof(int), 1, clayer->activeFP);
      PV_fseek(clayer->activeFP, fpos, SEEK_SET);
      status = intswritten == 1 ? PV_SUCCESS : PV_FAILURE;
   }
   else {
      status = PV_SUCCESS;
   }
   return status;
}

// copyDirect is never called.  Do we still need it?
/* copy src PVLayerCube to dest PVLayerCube */
/* initialize src, dest to beginning of data structures */
int copyDirect(pvdata_t * dest, pvdata_t * src, int nf, int nxSrc, int nySrc, int syDst, int sySrc)
{
   pvdata_t * to   = dest;
   pvdata_t * from = src;

   for (int j = 0; j < nySrc; j++) {
      to   = dest + j*syDst;
      from = src  + j*sySrc;
      for (int i = 0; i < nxSrc; i++) {
         for (int f = 0; f < nf; f++) {
            to[f] = from[f];
         }
         to   += nf;
         from += nf;
      }
   }
   return 0;
}

bool HyPerLayer::localDimensionsEqual(PVLayerLoc const * loc1, PVLayerLoc const * loc2) {
   return
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
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder = dest->loc.halo.up;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + topBorder*sy + leftBorder*sx;
   pvdata_t * dst0 = dest->data + (topBorder - 1)*sy + (leftBorder - 1)*sx;

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
   return 0;
}

int HyPerLayer::mirrorToNorth(PVLayerCube * dest, PVLayerCube * src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int nx = clayer->loc.nx;
   int nf = clayer->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder = dest->loc.halo.up;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + topBorder*sy + leftBorder*sx;
   pvdata_t * dst0 = dest->data + (topBorder-1)*sy + leftBorder*sx;

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
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + topBorder*sy + (nx + leftBorder - 1)*sx;
   pvdata_t * dst0 = dest->data + (topBorder-1)*sy + (nx + leftBorder)*sx;

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
   return 0;
}

int HyPerLayer::mirrorToWest(PVLayerCube* dest, PVLayerCube* src)
{
   if (!localDimensionsEqual(&dest->loc, &src->loc)) { return -1; }
   int ny = dest->loc.ny;
   int nf = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder = dest->loc.halo.up;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + topBorder*sy + leftBorder*sx;
   pvdata_t * dst0 = dest->data + topBorder*sy + (leftBorder - 1)*sx;

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
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + topBorder*sy + (nx + leftBorder - 1)*sx;
   pvdata_t * dst0 = dest->data + topBorder*sy + (nx + leftBorder)*sx;

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
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + (ny + topBorder - 1)*sy + leftBorder*sx;
   pvdata_t * dst0 = dest->data + (ny + topBorder)*sy + (leftBorder - 1)*sx;

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
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + (ny + topBorder -1)*sy + leftBorder*sx;
   pvdata_t * dst0 = dest->data + (ny + topBorder)*sy + leftBorder*sx;

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
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + (ny + topBorder - 1)*sy + (nx + leftBorder - 1)*sx;
   pvdata_t * dst0 = dest->data + (ny + topBorder)*sy + (nx + leftBorder)*sx;

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
   return 0;
}

/**
 * Return the label (if any) of a neuron in this layer.  A label may be the
 * orientation (for example) of a neuron.  Creating a label for a neuron is
 * normally done by offline analysis after the synaptic weights for connections
 * to the layer have been learned.
 */
int HyPerLayer::label(int k)
{
   if (labels == NULL) return 0;
   else                return labels[k];
}

int HyPerLayer::getNumMargin(){
   if (marginIndices == NULL){
      getMarginIndices();
   }
   return numMargin;
}

int * HyPerLayer::getMarginIndices(){
   if (marginIndices == NULL){
      int kMargin = 0;
      const PVLayerLoc * layerLoc = getLayerLoc();
      const int marginUp = layerLoc->halo.up;
      const int marginDn = layerLoc->halo.dn;
      const int marginLt = layerLoc->halo.lt;
      const int marginRt = layerLoc->halo.rt;
      numMargin = marginUp * marginDn * marginLt * marginRt;
      assert(numMargin == getNumExtended() - getNumNeurons());
      const int nf = layerLoc->nf;
      const int nx = layerLoc->nx;
      const int ny = layerLoc->ny;
      int nxExt = nx + marginRt + marginLt;
      int nyExt = ny + marginUp + marginDn;
      //int syExt = nf * nxExt;
      //int sxExt = nf;
      int * marginIndices = (int *) calloc(numMargin, sizeof(int));
      assert(marginIndices != NULL);
      // get North margin indices
      for (int kPreExt = 0; kPreExt < nf * nxExt * marginUp; kPreExt++) {
         marginIndices[kMargin++] = kPreExt;
      }
      assert(kMargin == nf * nxExt * marginUp);
      // get East margin indices
      for (int ky = marginUp; ky < marginUp + ny; ky++) {
         for (int kx = 0; kx < marginLt; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
               marginIndices[kMargin++] = kPreExt;
            }
         }
      }
      assert(kMargin == nf * nxExt * marginUp + nf * marginLt * ny);
      // get West margin indices
      for (int ky = marginUp; ky < marginUp + ny; ky++) {
         for (int kx = nx + marginLt; kx < nxExt; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               int kPreExt = kIndex(kx, ky, kf, nxExt, nyExt, nf);
               marginIndices[kMargin++] = kPreExt;
            }
         }
      }
      assert(kMargin == nf * nxExt * marginUp + nf * marginLt * ny + nf * marginUp * ny);
      // get South margin indices
      for (int kPreExt = kMargin; kPreExt < numMargin; kPreExt++) {
         marginIndices[kMargin++] = kPreExt;
      }
      assert(kMargin == numMargin);
   }
   return marginIndices;
}
// Template functions
//
template <typename T>
int HyPerLayer::copyFromBuffer(const T * buf, T * data,
      const PVLayerLoc * loc, bool extended, T scale)
{
   size_t sf, sx, sy;

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->halo.lt;
      nyBorder = loc->halo.up;
      sf = strideFExtended(loc);
      sx = strideXExtended(loc);
      sy = strideYExtended(loc);
   }
   else {
      sf = strideF(loc);
      sx = strideX(loc);
      sy = strideY(loc);
   }

   int ii = 0;
   for (int j = 0; j < ny; j++) {
      int jex = j + nyBorder;
      for (int i = 0; i < nx; i++) {
         int iex = i + nxBorder;
         for (int f = 0; f < nf; f++) {
            data[iex*sx + jex*sy + f*sf] = scale * buf[ii++];
         }
      }
   }
   return 0;
}


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


