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
#include "../connections/HyPerConn.hpp"
#include "../connections/TransposeConn.hpp"
#include "InitV.hpp"
#include "../io/fileio.hpp"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>
#include "Retina.hpp" // Only needed for the warning regarding spikingFlag; this #include can be removed when the warning is removed.

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
   this->margin = 0;
   this->numProbes = 0;
   this->ioAppend = 0;
   this->numChannels = 0;
   this->clayer = NULL;
   this->GSyn = NULL;
   this->labels = NULL;
   this->marginIndices = NULL;
   this->numMargin = 0;
   this->writeTime = 0;
   this->initialWriteTime = 0;
   this->phase = 0;

   this->initInfoCommunicatedFlag = false;
   this->dataStructuresAllocatedFlag = false;
   
   this->numSynchronizedMarginWidthLayers = 0;
   this->synchronizedMarginWidthLayers = NULL;

#ifdef PV_USE_OPENCL
   this->krUpdate = NULL;
   this->clV = NULL;
   this->clGSyn = NULL;
   this->clActivity = NULL;
   this->clPrevTime = NULL;
   this->clParams = NULL;
   this->numKernelArgs = 0;
   this->numEvents = 0;
   this->numWait = 0;
   this->evList = NULL;
   this->gpuAccelerateFlag=false;
#endif // PV_USE_OPENCL

   this->update_timer  = NULL;
   this->recvsyn_timer = NULL;
   this->publish_timer = NULL;
   this->io_timer      = NULL;

   return PV_SUCCESS;
}

///////
// Classes derived from HyPerLayer should call HyPerLayer::initialize themselves
// to take advantage of virtual methods.  Note that the HyPerLayer constructor
// does not call initialize.  This way, HyPerLayer::initialize can call virtual
// methods and the derived class's method will be the one that gets called.
int HyPerLayer::initialize(const char * name, HyPerCol * hc, int numChannels) {
   this->name = strdup(name);
   setParent(hc); // Could this line and the parent->addLayer line be combined in a HyPerLayer method?
   this->numChannels = numChannels;

   this->update_timer  = new Timer();
   this->recvsyn_timer = new Timer();
   this->publish_timer = new Timer();
   this->io_timer      = new Timer();

   PVParams * params = parent->parameters();

   int status = setParams(params);
   assert(status == PV_SUCCESS);

   writeActivityCalls = 0;
   writeActivitySparseCalls = 0;
   numDelayLevels = 1; // If a connection has positive delay so that more delay levels are needed, numDelayLevels is increased when HyPerConn::communicateInitInfo calls increaseDelayLevels
   maxRate = 1000.0f/parent->getDeltaTime();

   initClayer();

   // Moved to allocateDataStructures
   //   // If not mirroring, fill nonrestricted part of extended activity values
   //   // Since we calloc'ed, we only need to do this if valueBC is nonzero
   //   if (!useMirrorBCs() && getValueBC()!=0.0f) {
   //      const PVLayerLoc * loc = getLayerLoc();
   //      int nx = loc->nx;
   //      int ny = loc->ny;
   //      int nf = loc->nf;
   //      int nb = loc->nb;
   //      int idx = 0;
   //      for (int b=0; b<getLayerLoc()->nb; b++) {
   //         for(int k=0; k<(nx+2*nb)*nf; k++) {
   //            clayer->activity->data[idx] = getValueBC();
   //            idx++;
   //         }
   //      }
   //      for (int y=0; y<ny; y++) {
   //         for(int k=0; k<nb*nf; k++) {
   //            clayer->activity->data[idx] = getValueBC();
   //            idx++;
   //         }
   //         idx += nx;
   //         for(int k=0; k<nb*nf; k++) {
   //            clayer->activity->data[idx] = getValueBC();
   //            idx++;
   //         }
   //      }
   //      for (int b=0; b<getLayerLoc()->nb; b++) {
   //         for(int k=0; k<(nx+2*nb)*nf; k++) {
   //            clayer->activity->data[idx] = getValueBC();
   //            idx++;
   //         }
   //      }
   //      assert(idx==getNumExtended());
   //   }

   // must set ioAppend before addLayer is called (addLayer causes activity file to be opened using layerid)
   ioAppend = parent->getCheckpointReadFlag() ? 1 : 0;

   layerId = parent->addLayer(this);

   status = openOutputStateFile();

   // allocate storage for the input conductance arrays
   //
   // status = allocateBuffers(); // Moved to allocateDataStructures
   // assert(status == PV_SUCCESS);

   // // labels are not extended
   // labels = (int *) calloc(getNumNeurons(), sizeof(int));
   // assert(labels != NULL);

#ifdef PV_USE_OPENCL
   initUseGPUFlag();
#endif

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
   updateClayerMargin(clayer, margin);

   clayer->numNeurons  = loc->nx * loc->ny * loc->nf;

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

#ifdef PV_USE_OPENCL
//This method checks for a parameter telling Petavision to GPU accellerate
//this layer
void HyPerLayer::initUseGPUFlag() {
   PVParams * params = parent->parameters();
   assert(!params->presentAndNotBeenRead(name,"GPUAccelerate"));
   copyDataStoreFlag=false;
}

//this method sets up GPU related variables and calls the
//initializeThreadBuffers and initializeThreadKernels
int HyPerLayer::initializeGPU() {
   CLDevice * device = parent->getCLDevice();

   //copyToDevice=false;
   numWait = 0;
   numEvents = getNumCLEvents();
   evList = (cl_event *) malloc(numEvents*sizeof(cl_event));
   assert(evList != NULL);

   // TODO - fix to use device and layer parameters
   if (device->id() == 1) {
      nxl = 1;  nyl = 1;
   }
   else {
      nxl = 16; nyl = 8;
   }

   const char * kernel_name = getKernelName();
   initializeThreadBuffers(kernel_name);
   initializeThreadKernels(kernel_name);

   return PV_SUCCESS;
}
#endif

HyPerLayer::~HyPerLayer()
{
   if (parent->columnId() == 0) {
      printf("%32s: total time in %6s %10s: ", name, "layer", "recvsyn");
      recvsyn_timer->elapsed_time();
      printf("%32s: total time in %6s %10s: ", name, "layer", "update ");
      update_timer->elapsed_time();
      printf("%32s: total time in %6s %10s: ", name, "layer", "publish");
      publish_timer->elapsed_time();
      printf("%32s: total time in %6s %10s: ", name, "layer", "io     ");
      io_timer->elapsed_time();
      fflush(stdout);
   }
   delete recvsyn_timer;  recvsyn_timer = NULL;
   delete update_timer;   update_timer  = NULL;
   delete publish_timer;  publish_timer = NULL;
   delete io_timer;       io_timer      = NULL;

   freeClayer();
   free(name); name = NULL;
   freeChannels();

#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      delete krUpdate;
      delete clV;
      delete clActivity;
      delete clPrevTime;
      delete clParams;

      //   if (clGSyn != NULL) {
      //      for (int m = 0; m < numChannels; m++) {
      //         delete clGSyn[m];
      //      }
      //      free(clGSyn);
      //      clGSyn = NULL;
      //   }

      free(evList);
      evList = NULL;
   }

#endif

   free(labels); labels = NULL;
   free(marginIndices); marginIndices = NULL;
   for (int i_probe = 0; i_probe < this->numProbes; i_probe++){
      delete probes[i_probe];
   }
   free(probes);

   free(synchronizedMarginWidthLayers);
}

int HyPerLayer::freeClayer() {
   pvcube_delete(clayer->activity);

   if (clayer->activeFP != NULL) {
      PV_fclose(clayer->activeFP);
      clayer->activeFP = NULL;
   }

   free(clayer->activeIndices); clayer->activeIndices = NULL;
   free(clayer->prevActivity);  clayer->prevActivity = NULL;
   //free(clayer->activeIndices); clayer->activeIndices = NULL;
   free(clayer->V);             clayer->V = NULL;
   free(clayer);                clayer = NULL;

   return PV_SUCCESS;
}

void HyPerLayer::freeChannels()
{
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      //      for (int m = 0; m < numChannels; m++) {
      //         delete clGSyn[m];
      //      }
      //free(clGSyn);
      delete clGSyn;
      clGSyn = NULL;
   }
#endif

   // Test on GSyn because numChannels gets set in initialize, but GSyn gets allocated in allocateDataStructures, and only if numChannels>0.
   if (GSyn) {
      assert(numChannels>0);
      free(GSyn[0]);  // conductances allocated contiguously so frees all buffer storage
      free(GSyn);     // this frees the array pointers to separate conductance channels
      GSyn = NULL;
      numChannels = 0;
   }
}

#ifdef PV_USE_OPENCL
#endif

int HyPerLayer::allocateClayerBuffers() {
   int k;
   // clayer fields numNeurons, numExtended, loc, xScale, yScale, dx, dy, xOrigin, yOrigin were set in initClayer().
   assert(clayer);
   clayer->params = NULL;

   clayer->numActive = 0;

   int status = PV_SUCCESS;

   int statusV = allocateV();                      if (statusV!=PV_SUCCESS) status = PV_FAILURE;
   int statusA = allocateActivity();               if (statusA!=PV_SUCCESS) status = PV_FAILURE;
   int statusActIndices = allocateActiveIndices(); if (statusActIndices!=PV_SUCCESS) status = PV_FAILURE;
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

int HyPerLayer::allocateActiveIndices() {
   return allocateBuffer(&clayer->activeIndices, getNumNeurons(), "active indices");
}

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

   // nb and halo are set in calls to updateClayerMargin
   layerLoc->nb = 0; // margin;

   layerLoc->halo.lt = 0; // margin;
   layerLoc->halo.rt = 0; // margin;
   layerLoc->halo.dn = 0; // margin;
   layerLoc->halo.up = 0; // margin;

   return 0;
}

int HyPerLayer::updateClayerMargin(PVLayer * clayer, int new_margin) {
   // Only be called before buffers are allocated
   assert(clayer!=NULL);
   PVLayerLoc * loc = &clayer->loc;
   clayer->loc.nb = new_margin;
   PVHalo * halo = &loc->halo;
   halo->lt = new_margin;
   halo->rt = new_margin;
   halo->dn = new_margin;
   halo->up = new_margin;

   clayer->numExtended = (loc->nx+2*new_margin)*(loc->ny+2*new_margin)*loc->nf;
   return PV_SUCCESS;
}

int HyPerLayer::allocateBuffers() {
   // allocate memory for the input conductance arrays.
   // virtual so that subclasses can initialize additional buffers if needed.
   // Typically an overriding allocateBuffers should call HyPerLayer::allocateBuffers
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
   assert(!params->presentAndNotBeenRead(name, "restart"));
   // readRestart(params);
   if( restartFlag ) {
      double timef;
      status = readState(&timef);
   }
   else {
      if (this->getV()!=NULL) {
         InitV * initVObject = new InitV(parent, name);
         if( initVObject == NULL ) {
            fprintf(stderr, "HyPerLayer::initializeState error: layer %s unable to create InitV object\n", name);
            abort();
         }
         status = initVObject->calcV(this);
         delete initVObject;
         setActivity();
         if (status == PV_SUCCESS) status = updateActiveIndices();
      }
   }
   return status;
}

int HyPerLayer::setParams(PVParams * inputParams)
{
   readNxScale(inputParams);
   readNyScale(inputParams);
   readNf(inputParams);
   readMarginWidth(inputParams);
   readWriteStep(inputParams);
   readPhase(inputParams);
   readWriteSparseActivity(inputParams);
   readMirrorBCFlag(inputParams);
   readValueBC(inputParams);
   readRestart(inputParams);
#ifdef PV_USE_OPENCL
   readGPUAccelerate(inputParams);
#endif // PV_USE_OPENCL

   return PV_SUCCESS;
}

void HyPerLayer::readNxScale(PVParams * params) {
   nxScale = params->value(name, "nxScale", nxScale);
}

void HyPerLayer::readNyScale(PVParams * params) {
   nyScale = params->value(name, "nyScale", nyScale);
}

void HyPerLayer::readNf(PVParams * params) {
   numFeatures = (int) params->value(name, "nf", numFeatures);
}

void HyPerLayer::readMarginWidth(PVParams * params) {
   if (params->present(name, "marginWidth")) {
      margin = (int) params->value(name, "marginWidth");
      if (parent->columnId()==0) {
         fprintf(stderr, "HyPerLayer \"%s\": margins are adjusted automatically; parameter marginWidth is deprecated.\n", name);
      }
   }
}

void HyPerLayer::readWriteStep(PVParams * params) {
   writeStep = params->value(name, "writeStep", parent->getDeltaTime());
   if (writeStep>=0.0f) {
      readInitialWriteTime(params);
      writeTime = initialWriteTime-writeStep;
   }

}

void HyPerLayer::readInitialWriteTime(PVParams * params) {
   initialWriteTime = params->value(name, "initialWriteTime", parent->simulationTime());
}

void HyPerLayer::readPhase(PVParams * params) {
   phase = params->value(name, "phase", phase, true);
   if (phase<0) {
      if (parent->columnId()==0) fprintf(stderr, "Error in layer \"%s\": phase must be >= 0 (given value was %d).\n", name, phase);
      abort();
   }
}

void HyPerLayer::readWriteSparseActivity(PVParams * params) {
   writeSparseActivity = (bool) params->value(name, "writeSparseActivity", 0);
}

void HyPerLayer::readMirrorBCFlag(PVParams * params) {
   mirrorBCflag = (bool) params->value(name, "mirrorBCflag", mirrorBCflag);
}

void HyPerLayer::readValueBC(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "mirrorBCflag"));
   if (!mirrorBCflag) {
      valueBC = (pvdata_t) params->value(name, "valueBC", 0.0);
   }
}

void HyPerLayer::readRestart(PVParams * params) {
   restartFlag = params->value(name, "restart", 0.0f) != 0.0f;
}

#ifdef PV_USE_OPENCL
void HyPerLayer::readGPUAccelerate(PVParams * params) {
   gpuAccelerateFlag = params->value(name, "GPUAccelerate", gpuAccelerateFlag);
}

/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int HyPerLayer::initializeThreadBuffers(const char * kernel_name)
{
   int status = CL_SUCCESS;

   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
   const size_t size_ex = getNumExtended() * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //
   clV = NULL;
   if (clayer->V != NULL) {
      clV = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, clayer->V);
   }
   clActivity = device->createBuffer(CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size_ex, clayer->activity->data);
   clPrevTime = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->prevActivity);

   // defer creation of clParams to derived classes (as it is class specific)
   clParams = NULL;

   const size_t size_gsyn=getNumNeurons()*numChannels*sizeof(pvdata_t);
   //clGSyn = NULL;
   clGSyn = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_gsyn, GSyn[0]);
   //   if (numChannels > 0) {
   //      clGSyn = (CLBuffer **) malloc(numChannels*sizeof(CLBuffer *));
   //      assert(clGSyn != NULL);
   //
   //      for (int m = 0; m < numChannels; m++) {
   //         clGSyn[m] = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, GSyn[m]);
   //      }
   //   }

   return status;
}

int HyPerLayer::initializeThreadKernels(const char * kernel_name)
{
   // No kernels for base functionality for now
   return PV_SUCCESS;
}
#endif

int HyPerLayer::initFinish()
{
   return 0;
}

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
   int status = PV_SUCCESS;

   return status;
}

int HyPerLayer::openOutputStateFile() {
   char filename[PV_PATH_MAX];
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
      PV_Stream * pvstream = PV_fopen(filename,"r");
      if (pvstream) {
         int params[NUM_BIN_PARAMS];
         int numread = PV_fread(params, sizeof(int), NUM_BIN_PARAMS, pvstream);
         if (numread==NUM_BIN_PARAMS) {
            if (writeSparseActivity) {
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
   clayer->activeFP = pvp_open_write_file(filename, icComm, ioAppend);

   return PV_SUCCESS;
}

void HyPerLayer::synchronizeMarginWidth(HyPerLayer * layer) {
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

   int thisnb = this->getLayerLoc()->nb;
   int thatnb = layer->getLayerLoc()->nb;
   int result = 0;
   int status = PV_SUCCESS;
   if (thisnb < thatnb) {
      this->requireMarginWidth(thatnb, &result);
      if (result != thatnb) status = PV_FAILURE;
   }
   else if (thisnb > thatnb) {
      int result = thatnb;
      layer->requireMarginWidth(thisnb, &result);
      if (result != thisnb) status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      fprintf(stderr, "%s \"%s\" error in rank %d process: unable to synchronize margin width with layer \"%s\"\n", parent->parameters()->groupKeywordFromName(name), name, parent->columnId(), layer->getName());;
      exit(EXIT_FAILURE);
   }
   assert(this->getLayerLoc()->nb == layer->getLayerLoc()->nb);

   return;
}

int HyPerLayer::allocateDataStructures()
{
   // Once initialize and communicateInitInfo have been called, HyPerLayer has the
   // information it needs to allocate the membrane potential buffer V, the
   // activity buffer activity->data, and the data store.
   int status = PV_SUCCESS;

   allocateClayerBuffers();

   // If not mirroring, fill the boundaries with the value in the valueBC param
   if (!useMirrorBCs() && getValueBC()!=0.0f) {
      const PVLayerLoc * loc = getLayerLoc();
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int nb = loc->nb;
      int idx = 0;
      for (int b=0; b<getLayerLoc()->nb; b++) {
         for(int k=0; k<(nx+2*nb)*nf; k++) {
            clayer->activity->data[idx] = getValueBC();
            idx++;
         }
      }
      for (int y=0; y<ny; y++) {
         for(int k=0; k<nb*nf; k++) {
            clayer->activity->data[idx] = getValueBC();
            idx++;
         }
         idx += nx;
         for(int k=0; k<nb*nf; k++) {
            clayer->activity->data[idx] = getValueBC();
            idx++;
         }
      }
      for (int b=0; b<getLayerLoc()->nb; b++) {
         for(int k=0; k<(nx+2*nb)*nf; k++) {
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

int HyPerLayer::requireMarginWidth(int marginWidthNeeded, int * marginWidthResult) {
   if (margin < marginWidthNeeded) {
      assert(clayer);
      if (parent->columnId()==0) {
         printf("Layer \"%s\": adjusting margin width from %d to %d\n", name, margin, marginWidthNeeded);
      }
      margin = marginWidthNeeded;
      updateClayerMargin(clayer, margin);
   }
   *marginWidthResult = margin;
   if (synchronizedMarginWidthLayers != NULL) {
      for (int k=0; k<numSynchronizedMarginWidthLayers; k++) {
         HyPerLayer * l = synchronizedMarginWidthLayers[k];
         if (l->getLayerLoc()->nb < marginWidthNeeded) {
            synchronizedMarginWidthLayers[k]->requireMarginWidth(marginWidthNeeded, marginWidthResult);
            assert(*marginWidthResult == marginWidthNeeded);
         }
         assert(l->getLayerLoc()->nb == getLayerLoc()->nb);
      }
   }
   return PV_SUCCESS;
}

int HyPerLayer::requireChannel(int channelNeeded, int * numChannelsResult) {
   // TODO - set numChannels based on calls to requireChannel calls, and not have a numChannels argument in the constructors
   *numChannelsResult = numChannels;
   int status = channelNeeded < numChannels ? PV_SUCCESS : PV_FAILURE;
   if (status != PV_SUCCESS){
      fprintf(stderr, "Layer \"%s\": Channel %d does not exist, last allowable channel index is %d.\n", name, channelNeeded, numChannels-1);
   }
   return status;
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

#ifdef PV_USE_OPENCL
size_t HyPerLayer::getLayerDataStoreOffset(int delay)
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   size_t offset  = store->bufferOffset(LOCAL, delay);
   // (Rasmussen) still sorting this out
   // size_t offset2 = (store->bufferOffset(0, 0) - store->bufferOffset(LOCAL, delay));
   return offset;
}

int HyPerLayer::copyDataStoreCLBuffer() {
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   return store->copyBufferToDevice();
}
int HyPerLayer::waitForDataStoreCopy() {
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   return store->waitForCopy();
}

CLBuffer * HyPerLayer::getLayerDataStoreCLBuffer()
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   return store->getCLBuffer();
}

//int HyPerLayer::initializeDataStoreThreadBuffers()
//{
//   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
//   int status= store->initializeThreadBuffers(parent);
//   //status |= store->getCLBuffer()->copyToDevice(evCopyDataStore);
//   return status;
//}

#endif


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
   const int nxBorder = clayer->loc.nb;
   const int nyBorder = clayer->loc.nb;

   switch (borderId) {
   case 0:
      numNeurons = clayer->numNeurons;         break;
   case NORTHWEST:
      numNeurons = nxBorder * nyBorder * nf;   break;
   case NORTH:
      numNeurons = nx       * nyBorder * nf;   break;
   case NORTHEAST:
      numNeurons = nxBorder * nyBorder * nf;   break;
   case WEST:
      numNeurons = nxBorder * ny       * nf;   break;
   case EAST:
      numNeurons = nxBorder * ny       * nf;   break;
   case SOUTHWEST:
      numNeurons = nxBorder * nyBorder * nf;   break;
   case SOUTH:
      numNeurons = nx       * nyBorder * nf;   break;
   case SOUTHEAST:
      numNeurons = nxBorder * nyBorder * nf;   break;
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
   assert( cube->loc.nx == border->loc.nx );
   assert( cube->loc.ny == border->loc.ny );
   assert( cube->loc.nf == border->loc.nf );

   switch (whichBorder) {
   case NORTHWEST:
      return mirrorToNorthWest(border, cube);
   case NORTH:
      return mirrorToNorth(border, cube);
   case NORTHEAST:
      return mirrorToNorthEast(border, cube);
   case WEST:
      return mirrorToWest(border, cube);
   case EAST:
      return mirrorToEast(border, cube);
   case SOUTHWEST:
      return mirrorToSouthWest(border, cube);
   case SOUTH:
      return mirrorToSouth(border, cube);
   case SOUTHEAST:
      return mirrorToSouthEast(border, cube);
   default:
      fprintf(stderr, "ERROR:HyPerLayer:copyToBorder: bad border index %d\n", whichBorder);
      return -1;
   }

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

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->nb;
      nyBorder = loc->nb;
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
   int nxBorder, nyBorder;
   int numItems;

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   if (extended) {
      nxBorder = loc->nb;
      nyBorder = loc->nb;
      sf = strideFExtended(loc);
      sx = strideXExtended(loc);
      sy = strideYExtended(loc);
      numItems = nf*(nx+2*nxBorder)*(ny+2*nyBorder);
   }
   else {
      nxBorder = 0;
      nyBorder = 0;
      sf = strideF(loc);
      sx = strideX(loc);
      sy = strideY(loc);
      numItems = nf*nx*ny;
   }

   int ii = 0;
   for (int j = 0; j < ny; j++) {
      int jex = j + nyBorder;
      for (int i = 0; i < nx; i++) {
         int iex = i + nxBorder;
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

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->nb;
      nyBorder = loc->nb;
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
            data[iex*sx + jex*sy + f*sf] = scale * (pvdata_t) buf[ii++];
         }
      }
   }
   return 0;
}

int HyPerLayer::updateState(double timef, double dt) {
   int status;
   pvdata_t * gSynHead = GSyn==NULL ? NULL : GSyn[0];
   status = doUpdateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(),
         getNumChannels(), gSynHead, getSpikingFlag(), getCLayer()->activeIndices,
         &getCLayer()->numActive);
   if(status == PV_SUCCESS) status = updateActiveIndices();
   return status;
}


int HyPerLayer::resetGSynBuffers(double timef, double dt) {
   int status = PV_SUCCESS;
   if (GSyn == NULL) return PV_SUCCESS;
   resetGSynBuffers_HyPerLayer(this->getNumNeurons(), getNumChannels(), GSyn[0]); // resetGSynBuffers();
   return status;
}


int HyPerLayer::doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active)
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
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->nb);
   // moved to separate method to allow HyPerCol to control calling sequence
   //resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead); // resetGSynBuffers();

   return PV_SUCCESS;
}

int HyPerLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_HyPerLayer(getNumNeurons(), clayer->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb);
}

int HyPerLayer::updateBorder(double time, double dt)
{
   int status = PV_SUCCESS;

#ifdef PV_USE_OPENCL
   // wait for memory to be copied from device
   if (numWait > 0) {
      status |= clWaitForEvents(numWait, evList);
   }
   for (int i = 0; i < numWait; i++) {
      clReleaseEvent(evList[i]);
   }
   numWait = 0;

   //   status |= clWaitForEvents(1, &evUpdate);
   //   clReleaseEvent(evUpdate);
#endif

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

int HyPerLayer::updateActiveIndices() {
   if( writeSparseActivity ) return calcActiveIndices(); else return PV_SUCCESS;
}

int HyPerLayer::calcActiveIndices() {
   int numActive = 0;
   PVLayerLoc & loc = clayer->loc;
   pvdata_t * activity = clayer->activity->data;

   for (int k = 0; k < getNumNeurons(); k++) {
      const int kex = kIndexExtended(k, loc.nx, loc.ny, loc.nf, loc.nb);
      if (activity[kex] > 0.0) {
         clayer->activeIndices[numActive++] = globalIndexFromLocal(k, loc);
      }
   }
   clayer->numActive = numActive;
   return PV_SUCCESS;
}

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


//int HyPerLayer::setActivity() {
//   const int nx = getLayerLoc()->nx;
//   const int ny = getLayerLoc()->ny;
//   const int nf = getLayerLoc()->nf;
//   const int nb = getLayerLoc()->nb;
//   pvdata_t * activity = getCLayer()->activity->data;
//   pvdata_t * V = getV();
//   for( int k=0; k<getNumExtended(); k++ ) {
//      activity[k] = 0; // Would it be faster to only do the margins?
//   }
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      int kex = kIndexExtended(k, nx, ny, nf, nb);
//      activity[kex] = V[k];
//   }
//   return PV_SUCCESS;
//}

//int HyPerLayer::resetGSynBuffers() {
//   int n = getNumNeurons();
//   for( int k=0; k<numChannels; k++ ) {
//      resetBuffer( getChannel((ChannelType) k), n );
//   }
//   // resetBuffer( getChannel(CHANNEL_EXC), n );
//   // resetBuffer( getChannel(CHANNEL_INH), n );
//   return PV_SUCCESS;
//}
//
//int HyPerLayer::resetBuffer( pvdata_t * buf, int numItems ) {
//   assert(buf);
//   for( int k=0; k<numItems; k++ ) buf[k] = 0.0;
//   return PV_SUCCESS;
//}
//

int HyPerLayer::recvAllSynapticInput() {
   int status = PV_SUCCESS;
   int numConnections = parent->numberOfConnections();
   for (int c=0; c<numConnections; c++) {
      HyPerConn * conn = parent->getConnection(c);
      if (conn->postSynapticLayer()!=this) continue;
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
            status = recvSynapticInput(conn, &cube, arbor);
         }
         else{
            //Source layer is pre layer in current connection, post layer in original connection
            //Target layer is post layer in current connection, pre layer in original connection
            //cube is activity buffer of source layer
            //conn is source to target
            status = recvSynapticInputFromPost(conn, &cube, arbor);
         }
         assert(status == PV_SUCCESS || status == PV_BREAK);
         if (status == PV_BREAK){
            break;
         }
      }
   }
   return status;
}

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
   //Cast to transpose conn
   TransposeConn * sourceToTargetConn = dynamic_cast <TransposeConn*> (conn);
   if(sourceToTargetConn == NULL){
      fprintf(stderr, "HyPerLayer \"%s\": Updating GSyn buffer from post perspective requires connection %s to be a TransposeConn.\n", name, conn->getName());
      abort();
   }
   //update conn to original connection
   HyPerConn * targetToSourceConn = sourceToTargetConn->getOriginalConn();
   // Don't need TransposeConn to have the same pre and post as originalConn but flipped.  nx,ny,nf must be consistent, but that's checked in initialization.
   // //Assert that the transpose is opposite of the original connection
   // if(targetToSourceConn->preSynapticLayer()->getLayerId() != sourceToTargetConn->postSynapticLayer()->getLayerId() ||
   //    targetToSourceConn->postSynapticLayer()->getLayerId() != sourceToTargetConn->preSynapticLayer()->getLayerId()){
   //    fprintf(stderr, "HyPerLayer \"%s\": Transpose connection %s must be the same connection in the oposite direction of %s.\n", name, sourceToTargetConn->getName(), conn->getName());
   //    abort();
   // }

   recvsyn_timer->start();

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

   const PVLayerLoc * sourceLoc = sourceToTargetConn->preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * targetLoc = getLayerLoc();
   for (int kTargetRes = 0; kTargetRes < numRestricted; kTargetRes++){
      //Change restricted to extended post neuron
      int kTargetExt = kIndexExtended(kTargetRes, targetLoc->nx, targetLoc->ny, targetLoc->nf, targetLoc->nb);
      bool inWindow; 
      inWindow = inWindowExt(arborID, kTargetExt);
      if(!inWindow) continue;
      //
      //Get start index of source from gsyn in restricted
      // We have to use gSynPatchStart instead of aPostOffset because originalConn's post-synaptic layer's nb may not be the same as conn's pre-layer's nb.
      int sourceRes = targetToSourceConn->getGSynPatchStart(kTargetExt, arborID);
      int sourceExt= kIndexExtended(sourceRes, sourceLoc->nx, sourceLoc->ny, sourceLoc->nf, sourceLoc->nb);
      int sourceXExt = kxPos(sourceExt, sourceLoc->nx + 2*sourceLoc->nb, sourceLoc->ny + 2*sourceLoc->nb, sourceLoc->nf);
      int sourceYExt = kyPos(sourceExt, sourceLoc->nx + 2*sourceLoc->nb, sourceLoc->ny + 2*sourceLoc->nb, sourceLoc->nf);
      int sourceF = featureIndex(sourceExt, sourceLoc->nx + 2*sourceLoc->nb, sourceLoc->ny + 2*sourceLoc->nb, sourceLoc->nf);

      //Grab patch given the post
      PVPatch * shrunkenWeights = targetToSourceConn->getWeights(kTargetExt, arborID);
      //Grab offset
      int offset = shrunkenWeights->offset;
      //Get x and y in patch space
      //conn is target to source
      int patchX = kxPos(offset, targetToSourceConn->xPatchSize(), targetToSourceConn->yPatchSize(), targetToSourceConn->fPatchSize());
      int patchY = kyPos(offset, targetToSourceConn->xPatchSize(), targetToSourceConn->yPatchSize(), targetToSourceConn->fPatchSize());

      //Move source X and Y to offset
      sourceXExt -= patchX; 
      sourceYExt -= patchY; 

      //Change sourceExt back to extended source index, but unshrunken
      int startSourceExt = kIndex(sourceXExt, sourceYExt, sourceF, sourceLoc->nx + 2*sourceLoc->nb, sourceLoc->ny + 2*sourceLoc->nb, sourceLoc->nf);

      //Calculate target's start of gsyn
      pvdata_t * gSynPatchHead = this->getChannel(sourceToTargetConn->getChannel());
      pvdata_t * gSynPatchPos = gSynPatchHead + kTargetRes;

      //get source layer's extended y stride
      int sy  = (sourceLoc->nx+2*sourceLoc->nb)*sourceLoc->nf;
      //get source layer's patch y stride
      int syp = targetToSourceConn->yPatchStride(); // Should be correct even if targetToSourceConn points to a different layer than sourceToTargetConn's pre.
      //Iterate through y patch
      int numPerStride = targetToSourceConn->xPatchSize() * targetToSourceConn->fPatchSize();
      const PVLayerLoc * origPostLoc = targetToSourceConn->postSynapticLayer()->getLayerLoc();
      int kTargetOrigConnExt = kIndexExtended(kTargetRes, origPostLoc->nx, origPostLoc->ny, origPostLoc->nf, origPostLoc->nb);
      int kernelIndex = targetToSourceConn->patchToDataLUT(kTargetOrigConnExt);
      uint4 * rngPtr = conn->getRandState(kTargetRes);
      for (int ky = 0; ky < targetToSourceConn->yPatchSize(); ky++){
         float * activityY = &(activity->data[startSourceExt + ky*sy]);
         float * weightY = targetToSourceConn->get_wDataHead(arborID, kernelIndex) + ky*syp;
         (conn->accumulateFunctionFromPostPointer)(numPerStride, gSynPatchPos, activityY, weightY, dt_factor, rngPtr);
      }
   }
   recvsyn_timer->stop();
   return PV_SUCCESS;
}

/**
 * Receive synaptic input from pre synaptic layer by looping over pre synaptic neurons 
 */
int HyPerLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID)
{
   recvsyn_timer->start();

   assert(arborID >= 0);
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
   fflush(stdout);
#endif // DEBUG_OUTPUT

   float dt_factor = getConvertToRateDeltaTimeFactor(conn);

   for (int kPre = 0; kPre < numExtended; kPre++) {

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
      if (a == 0.0f) continue;  // TODO - assume activity is sparse so make this common branch

      PVPatch * weights = conn->getWeights(kPre, arborID);

      // WARNING - assumes weight and GSyn patches from task same size
      //         - assumes patch stride sf is 1

      int nk  = conn->fPatchSize() * weights->nx;
      int ny  = weights->ny;
      int sy  = conn->getPostNonextStrides()->sy;       // stride in layer
      int syw = conn->yPatchStride();                   // stride in patch
      pvdata_t * gSynPatchHead = this->getChannel(conn->getChannel());
      size_t gSynPatchStartIndex = conn->getGSynPatchStart(kPre, arborID);
      pvdata_t * gSynPatchStart = gSynPatchHead + gSynPatchStartIndex;
      // GTK: gSynPatchStart redefined as offset from start of gSyn buffer
      // TODO - unroll
      pvdata_t * data = conn->get_wData(arborID,kPre);
      uint4 * rngPtr = conn->getRandState(kPre);
      for (int y = 0; y < ny; y++) {
         (conn->accumulateFunctionPointer)(nk, gSynPatchStart + y*sy, a, data + y*syw, rngPtr);
      }
   }

   recvsyn_timer->stop();

   return PV_SUCCESS;
}

#ifdef OBSOLETE // Marked obsolete July 25, 2013.  recvSynapticInput is now called by recvAllSynapticInput, called by HyPerCol, so deliver andtriggerReceive aren't needed.
int HyPerLayer::triggerReceive(InterColComm* comm)
{
   // deliver calls recvSynapticInput for all connections for which this layer is presynaptic (i.e. all connections made by this layer)
   //
   int status = comm->deliver(parent, getLayerId());
   //#ifdef PV_USE_OPENCL
   //   if((gpuAccelerateFlag)&&(copyToDevice)) {
   //      status |= getChannelCLBuffer()->copyToDevice(&evList[getEVGSyn()]);
   ////      status |= getChannelCLBuffer()->copyToDevice(&evList[getEVGSynE()]);
   ////      status |= getChannelCLBuffer()->copyToDevice(&evList[getEVGSynI()]);
   //      //numWait += 2;
   //      numWait ++;
   //   }
   //#endif
   return status;
}
#endif // OBSOLETE

int HyPerLayer::publish(InterColComm* comm, double time)
{
   publish_timer->start();

   if ( useMirrorBCs() ) {
      for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
         mirrorInteriorToBorder(borderId, clayer->activity, clayer->activity);
      }
   }

   int status = comm->publish(this, clayer->activity);
#ifdef PV_USE_OPENCL
   if(copyDataStoreFlag) {
      status |= copyDataStoreCLBuffer();
      //status |= getLayerDataStoreCLBuffer()->copyToDevice(evCopyDataStore);
      //numWait += 1;
   }
#endif

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

int HyPerLayer::outputState(double timef, bool last)
{
   int status = PV_SUCCESS;

   io_timer->start();

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(timef);
   }


   if (timef >= writeTime && writeStep >= 0) {
      writeTime += writeStep;
      if (writeSparseActivity) {
         status = writeActivitySparse(timef);
      }
      else {
         status = writeActivity(timef);
      }
   }

   io_timer->stop();
   return status;
}

#ifdef OBSOLETE // Marked obsolete Dec 18, 2012.  Nothing calls this function (probably obsoleted by move to checkpointRead/Write
/**
 * Return a file name to be used for output file for layer data
 *
 * WARNING - assumes length of buf >= PV_PATH_MAX
 */
const char * HyPerLayer::getOutputFilename(char * buf, const char * dataName, const char * term)
{
   snprintf(buf, PV_PATH_MAX-1, "%s/%s_%s%s.pvp", parent->getOutputPath(), getName(), dataName, term);
   return buf;
}
#endif // OBSOLETE

int HyPerLayer::checkpointRead(const char * cpDir, double * timed) {
   InterColComm * icComm = parent->icCommunicator();
   char basepath[PV_PATH_MAX];
   char filename[PV_PATH_MAX];
   int lenbase = snprintf(basepath, PV_PATH_MAX, "%s/%s", cpDir, name);
   if (lenbase+strlen("_nextWrite.bin") >= PV_PATH_MAX) { // currently _nextWrite.bin is the longest suffix needed
      if (icComm->commRank()==0) {
         fprintf(stderr, "HyPerLayer::checkpointRead error in layer \"%s\".  Base pathname \"%s/%s_\" too long.\n", name, cpDir, name);
      }
      abort();
   }
   double filetime;
   assert(filename != NULL);
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_A.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   int status = readBufferFile(filename, icComm, &filetime, &clayer->activity->data, 1, /*extended*/true, getLayerLoc());
   assert(status == PV_SUCCESS);
   *timed = filetime;
   updateActiveIndices();

   if( getV() != NULL ) {
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s_V.pvp", basepath);
      assert(chars_needed < PV_PATH_MAX);
      pvdata_t * V = getV();
      status = readBufferFile(filename, icComm, &filetime, &V, 1, /*extended*/false, getLayerLoc());
      assert(status == PV_SUCCESS);
      if( filetime != *timed && parent->icCommunicator()->commRank() == 0 ) {
         fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, filetime, *timed);
      }
   }

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Delays.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   status = readDataStoreFromFile(filename, icComm, &filetime);
   assert(status == PV_SUCCESS);
   if( filetime != *timed && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, filetime, *timed);
   }

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
      if (writeSparseActivity) {
         nfname = "numframes_sparse";
         num_calls_ptr = &writeActivitySparseCalls;
      }
      else {
         nfname = "numframes";
         num_calls_ptr = &writeActivityCalls;
      }
      struct stat statbuffer;
      int statstatus[2];
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s_%s.bin", basepath, nfname);
      assert(chars_needed < PV_PATH_MAX);
      if (parent->columnId()==0) {
         statstatus[0] = stat(filename, &statbuffer);
         statstatus[1] = errno;
      }
      MPI_Bcast(statstatus, 2, MPI_INT, 0/*root*/, icComm->communicator());

      if (statstatus[0]==0) {
         parent->readScalarFromFile(cpDir, getName(), nfname, num_calls_ptr, 0);
      }
      else {
         if (statstatus[1] == ENOENT) {
            *num_calls_ptr = 0;
            if (icComm->commRank()==0) {
               fprintf(stderr, "checkpointRead warning: file \"%s\" not found; will use %d for the value.\n", filename, *num_calls_ptr);
            }
         }
         else {
            if (icComm->commRank()==0) {
               fprintf(stderr, "checkpointRead error determining status of file \"%s\": %s", filename, strerror(errno));
            }
            MPI_Barrier(icComm->communicator());
            exit(EXIT_FAILURE);
         }
      }
   }

   return PV_SUCCESS;
}

int HyPerLayer::readBufferFile(const char * filename, InterColComm * comm, double * timed, pvdata_t ** buffers, int numbands, bool extended, const PVLayerLoc * loc) {
   PV_Stream * readFile = pvp_open_read_file(filename, comm);
   int rank = comm->commRank();
   assert( (readFile != NULL && rank == 0) || (readFile == NULL && rank != 0) );
   int numParams = NUM_BIN_PARAMS;
   int params[NUM_BIN_PARAMS];
   int status = pvp_read_header(readFile, comm, params, &numParams);
   if (status != PV_SUCCESS) {
      read_header_err(filename, comm, numParams, params);
   }

   switch(params[INDEX_FILE_TYPE]) {
   case PVP_FILE_TYPE:
      *timed = timeFromParams(params);
      break;
   case PVP_ACT_FILE_TYPE:
      status = pvp_read_time(readFile, comm, 0/*root process*/, timed);
      if (status!=PV_SUCCESS) {
         fprintf(stderr, "HyPerLayer::readBufferFile error reading timestamp in file \"%s\"\n", filename);
         abort();
      }
      if (rank==0) {
         fprintf(stderr,"HyPerLayer::readBufferFile error: filename \"%s\" is compressed spiking file, but this filetype has not yet been implemented in this case.\n", filename);
      }
      status = PV_FAILURE;
      break;
   case PVP_NONSPIKING_ACT_FILE_TYPE:
      status = pvp_read_time(readFile, comm, 0/*root process*/, timed);
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
   return status;
}

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
      if (timeptr != NULL) {
         if (l==0) {
            *timeptr = tlevel;
         }
         else {
            if (tlevel != *timeptr && comm->commRank()==0) {
               fprintf(stderr, "Warning: timestamp on delay level %d does not agree with that of delay level 0 (%g versus %g).\n", l, tlevel, *timeptr);
            }
         }
      }
      pvdata_t * buffer = (pvdata_t *) datastore->buffer(0, l);
      int status1 = scatterActivity(readFile, comm, 0/*root process*/, buffer, getLayerLoc(), true);
      if (status1 != PV_SUCCESS) status = PV_FAILURE;
   }
   assert(status == PV_SUCCESS);
   pvp_close_file(readFile, comm);
   return status;
}

#ifdef OBSOLETE // Marked obsolete May 1, 2013.  Use HyPerCol template function readScalarFromFile instead
int HyPerLayer::readScalarFloat(const char * cp_dir, const char * val_name, double * val_ptr, double default_value) {
   int status = PV_SUCCESS;
   if( parent->icCommunicator()->commRank() == 0 ) {
      char filename[PV_PATH_MAX];
      int chars_needed;
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, getName(), val_name);
      if(chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerLayer::readScalarFloat error: path %s/%s_%s.bin is too long.\n", cp_dir, getName(), val_name);
         abort();
      }
      PV_Stream * pvstream = PV_fopen(filename, "r");
      *val_ptr = default_value;
      if (pvstream==NULL  && parent->icCommunicator()->commRank() == 0 ) {
         fprintf(stderr, "HyPerLayer::readScalarFloat warning: unable to open path %s for reading.  writeTime will be %f\n", filename, default_value);
      }
      else {
         int num_read = PV_fread(val_ptr, sizeof(*val_ptr), 1UL, pvstream);
         if (num_read != 1) {
            fprintf(stderr, "HyPerLayer::readScalarFloat warning: unable to read from %s.  writeTime will be %f\n", filename, default_value);
         }
      }
      PV_fclose(pvstream);
   }
#ifdef PV_USE_MPI
   MPI_Bcast(val_ptr, 1, MPI_DOUBLE, 0, getParent()->icCommunicator()->communicator());
#endif // PV_USE_MPI

   return status;
}
#endif // OBSOLETE

int HyPerLayer::checkpointWrite(const char * cpDir) {
   // Writes checkpoint files for V, A, and datastore to files in working directory
   InterColComm * icComm = parent->icCommunicator();
   char basepath[PV_PATH_MAX];
   char filename[PV_PATH_MAX];
   int lenbase = snprintf(basepath, PV_PATH_MAX, "%s/%s", cpDir, name);
   if (lenbase+strlen("_nextWrite.bin") >= PV_PATH_MAX) { // currently _nextWrite.bin is the longest suffix needed
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

   parent->writeScalarToFile(cpDir, getName(), "nextWrite", writeTime);

   if (parent->columnId()==0) {
      if (clayer->activeFP) {
         long activityfilepos = getPV_StreamFilepos(clayer->activeFP);
         parent->writeScalarToFile(cpDir, getName(), "filepos", activityfilepos);
      }
   }

   if (writeStep>=0.0f) {
      if (writeSparseActivity) {
         parent->writeScalarToFile(cpDir, getName(), "numframes_sparse", writeActivitySparseCalls);
      }
      else {
         parent->writeScalarToFile(cpDir, getName(), "numframes", writeActivityCalls);
      }
   }

   return PV_SUCCESS;
}

int HyPerLayer::writeBufferFile(const char * filename, InterColComm * comm, double timed, pvdata_t ** buffers, int numbands, bool extended, const PVLayerLoc * loc) {
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

#ifdef OBSOLETE // Marked obsolote April 23, 2013.  Use the template function writeScalarToFile instead
int HyPerLayer::writeScalarFloat(const char * cp_dir, const char * val_name, double val) {
   int status = PV_SUCCESS;
   if (parent->columnId()==0)  {
      char filename[PV_PATH_MAX];
      int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, name, val_name);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "writeScalarFloat error: path %s/%s_%s.bin is too long.\n", cp_dir, name, val_name);
         abort();
      }
      PV_Stream * pvstream = PV_fopen(filename, "w");
      if (pvstream==NULL) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error: unable to open path %s for writing.\n", filename);
         abort();
      }
      int num_written = PV_fwrite(&val, sizeof(val), 1, pvstream);
      if (num_written != 1) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error while writing to %s.\n", filename);
         abort();
      }
      PV_fclose(pvstream);
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.txt", cp_dir, name, val_name);
      assert(chars_needed < PV_PATH_MAX);
      pvstream = PV_fopen(filename, "w");
      if (pvstream==NULL) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error: unable to open path %s for writing.\n", filename);
         abort();
      }
      fprintf(pvstream->fp, "%f\n", val);
      PV_fclose(pvstream);
   }
   return status;
}
#endif // OBSOLETE

int HyPerLayer::readState(double * timef)
{
   char last_dir[PV_PATH_MAX];
   int chars_needed = snprintf(last_dir, PV_PATH_MAX, "%s/Last", parent->getOutputPath());
   if (chars_needed >= PV_PATH_MAX) {
      if (parent->icCommunicator()->commRank()==0) {
         fprintf(stderr, "HyPerLayer::initializeState error: path \"%s/Last\" too long.\n", parent->getOutputPath());
      }
      abort();
   }
   return checkpointRead(last_dir, timef);
}

int HyPerLayer::writeActivitySparse(double timed)
{
   int status = PV::writeActivitySparse(clayer->activeFP, parent->icCommunicator(), timed, clayer);
   incrementNBands(&writeActivitySparseCalls);
   return status;
}

// write non-spiking activity
int HyPerLayer::writeActivity(double timed)
{
   int status = PV::writeActivity(clayer->activeFP, parent->icCommunicator(), timed, clayer);
   incrementNBands(&writeActivityCalls);
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

int HyPerLayer::mirrorToNorthWest(PVLayerCube * dest, PVLayerCube * src)
{
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + nb*sy + nb*sx;
   pvdata_t * dst0 = dest->data + (nb - 1)*sy + (nb - 1)*sx;

   for (int ky = 0; ky < nb; ky++) {
      pvdata_t * to   = dst0 - ky*sy;
      pvdata_t * from = src0 + ky*sy;
      for (int kx = 0; kx < nb; kx++) {
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
   int nx = clayer->loc.nx;
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + nb*sy + nb*sx;
   pvdata_t * dst0 = dest->data + (nb-1)*sy + nb*sx;

   for (int ky = 0; ky < nb; ky++) {
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
   int nx = clayer->loc.nx;
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + nb*sy + (nx + nb - 1)*sx;
   pvdata_t * dst0 = dest->data + (nb-1)*sy + (nx + nb)*sx;

   for (int ky = 0; ky < nb; ky++) {
      pvdata_t * to   = dst0 - ky*sy;
      pvdata_t * from = src0 + ky*sy;
      for (int kx = 0; kx < nb; kx++) {
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
   int ny = clayer->loc.ny;
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + nb*sy + nb*sx;
   pvdata_t * dst0 = dest->data + nb*sy + (nb - 1)*sx;

   for (int ky = 0; ky < ny; ky++) {
      pvdata_t * to   = dst0 + ky*sy;
      pvdata_t * from = src0 + ky*sy;
      for (int kx = 0; kx < nb; kx++) {
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
   int nx = clayer->loc.nx;
   int ny = clayer->loc.ny;
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + nb*sy + (nx + nb - 1)*sx;
   pvdata_t * dst0 = dest->data + nb*sy + (nx + nb)*sx;

   for (int ky = 0; ky < ny; ky++) {
      pvdata_t * to   = dst0 + ky*sy;
      pvdata_t * from = src0 + ky*sy;
      for (int kx = 0; kx < nb; kx++) {
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
   int ny = clayer->loc.ny;
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + (ny + nb - 1)*sy + nb*sx;
   pvdata_t * dst0 = dest->data + (ny + nb)*sy + (nb - 1)*sx;

   for (int ky = 0; ky < nb; ky++) {
      pvdata_t * to   = dst0 + ky*sy;
      pvdata_t * from = src0 - ky*sy;
      for (int kx = 0; kx < nb; kx++) {
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
   int nx = clayer->loc.nx;
   int ny = clayer->loc.ny;
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + (ny + nb -1)*sy + nb*sx;
   pvdata_t * dst0 = dest->data + (ny + nb)*sy + nb*sx;

   for (int ky = 0; ky < nb; ky++) {
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
   int nx = clayer->loc.nx;
   int ny = clayer->loc.ny;
   int nf = clayer->loc.nf;
   int nb = dest->loc.nb;
   size_t sf = strideFExtended(&dest->loc);
   size_t sx = strideXExtended(&dest->loc);
   size_t sy = strideYExtended(&dest->loc);

   pvdata_t * src0 = src-> data + (ny + nb - 1)*sy + (nx + nb - 1)*sx;
   pvdata_t * dst0 = dest->data + (ny + nb)*sy + (nx + nb)*sx;

   for (int ky = 0; ky < nb; ky++) {
      pvdata_t * to   = dst0 + ky*sy;
      pvdata_t * from = src0 - ky*sy;
      for (int kx = 0; kx < nb; kx++) {
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
      nxBorder = loc->nb;
      nyBorder = loc->nb;
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

