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
*/

#include "HyPerLayer.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "checkpointing/CheckpointEntryRandState.hpp"
#include "columns/HyPerCol.hpp"
#include "connections/BaseConnection.hpp"
#include "connections/TransposeConn.hpp"
#include "include/default_params.h"
#include "include/pv_common.h"
#include "io/FileStream.hpp"
#include "io/io.hpp"
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string.h>

namespace PV {

// This constructor is protected so that only derived classes can call it.
// It should be called as the normal method of object construction by
// derived classes.  It should NOT call any virtual methods
HyPerLayer::HyPerLayer() { initialize_base(); }

HyPerLayer::HyPerLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

// initialize_base should be called only by constructors.  It should not
// call any virtual methods, because polymorphism is not available when
// a base class constructor is inherited from a derived class constructor.
// In general, initialize_base should be used only to initialize member variables
// to safe values.
int HyPerLayer::initialize_base() {
   name                         = NULL;
   probes                       = NULL;
   nxScale                      = 1.0f;
   nyScale                      = 1.0f;
   numFeatures                  = 1;
   mirrorBCflag                 = 0;
   xmargin                      = 0;
   ymargin                      = 0;
   numProbes                    = 0;
   numChannels                  = 2;
   clayer                       = NULL;
   GSyn                         = NULL;
   marginIndices                = NULL;
   numMargin                    = 0;
   writeTime                    = 0;
   initialWriteTime             = 0;
   triggerFlag                  = false; // Default to update every timestamp
   triggerLayer                 = NULL;
   triggerLayerName             = NULL;
   triggerBehavior              = NULL;
   triggerBehaviorType          = NO_TRIGGER;
   triggerResetLayerName        = NULL;
   triggerOffset                = 0;
   initializeFromCheckpointFlag = false;

   mLastUpdateTime  = 0.0;
   mLastTriggerTime = 0.0;

   phase = 0;

   numSynchronizedMarginWidthLayers = 0;
   synchronizedMarginWidthLayers    = NULL;

   dataType       = PV_FLOAT;
   dataTypeString = NULL;

#ifdef PV_USE_CUDA
   allocDeviceV             = false;
   allocDeviceGSyn          = false;
   allocDeviceActivity      = false;
   allocDeviceDatastore     = false;
   allocDeviceActiveIndices = false;
   d_V                      = NULL;
   d_GSyn                   = NULL;
   d_Activity               = NULL;
   d_Datastore              = NULL;
   d_ActiveIndices          = NULL;
   d_numActive              = NULL;
   updatedDeviceActivity    = true; // Start off always updating activity
   updatedDeviceDatastore   = true;
   updatedDeviceGSyn        = true;
   mRecvGpu                 = false;
   mUpdateGpu               = false;
   krUpdate                 = NULL;
#ifdef PV_USE_CUDNN
   cudnn_GSyn      = NULL;
   cudnn_Datastore = NULL;
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   update_timer    = NULL;
   recvsyn_timer   = NULL;
   publish_timer   = NULL;
   timescale_timer = NULL;
   io_timer        = NULL;

#ifdef PV_USE_CUDA
   gpu_recvsyn_timer = NULL;
   gpu_update_timer  = NULL;
#endif

   thread_gSyn = NULL;
   recvConns.clear();

   return PV_SUCCESS;
}

///////
/// Classes derived from HyPerLayer should call HyPerLayer::initialize themselves
/// to take advantage of virtual methods.  Note that the HyPerLayer constructor
/// does not call initialize.  This way, HyPerLayer::initialize can call virtual
/// methods and the derived class's method will be the one that gets called.
int HyPerLayer::initialize(const char *name, HyPerCol *hc) {
   int status = BaseLayer::initialize(name, hc);
   if (status != PV_SUCCESS) {
      return status;
   }

   PVParams *params = parent->parameters();

   status = readParams();
   assert(status == PV_SUCCESS);

   writeTime                = initialWriteTime;
   writeActivityCalls       = 0;
   writeActivitySparseCalls = 0;
   numDelayLevels = 1; // If a connection has positive delay so that more delay levels are needed,
   // numDelayLevels is increased when BaseConnection::communicateInitInfo calls
   // increaseDelayLevels

   initClayer();

   mLastUpdateTime  = parent->getDeltaTime();
   mLastTriggerTime = parent->getDeltaTime();
   return PV_SUCCESS;
}

int HyPerLayer::initClayer() {
   clayer     = (PVLayer *)calloc(1UL, sizeof(PVLayer));
   int status = PV_SUCCESS;
   if (clayer == NULL) {
      Fatal().printf(
            "HyPerLayer \"%s\" error in rank %d process: unable to allocate memory for Clayer.\n",
            name,
            parent->columnId());
   }

   PVLayerLoc *loc = &clayer->loc;
   setLayerLoc(loc, nxScale, nyScale, numFeatures, parent->getNBatch());
   assert(loc->halo.lt == 0 && loc->halo.rt == 0 && loc->halo.dn == 0 && loc->halo.up == 0);

   int nBatch = parent->getNBatch();

   clayer->numNeurons  = loc->nx * loc->ny * loc->nf;
   clayer->numExtended = clayer->numNeurons; // initially, margin is zero; it will be updated as
   // needed during the communicateInitInfo stage.
   clayer->numNeuronsAllBatches  = nBatch * loc->nx * loc->ny * loc->nf;
   clayer->numExtendedAllBatches = clayer->numNeuronsAllBatches;

   double xScaled = -log2((double)nxScale);
   double yScaled = -log2((double)nyScale);

   int xScale = (int)nearbyint(xScaled);
   int yScale = (int)nearbyint(yScaled);

   clayer->xScale = xScale;
   clayer->yScale = yScale;

   // Other fields of clayer will be set in allocateClayerBuffers, or during updateState
   return status;
}

HyPerLayer::~HyPerLayer() {
   delete recvsyn_timer;
   delete update_timer;
   delete publish_timer;
   delete timescale_timer;
   delete io_timer;
#ifdef PV_USE_CUDA
   delete gpu_recvsyn_timer;
   delete gpu_update_timer;
#endif

   delete mOutputStateStream;

   delete mInitVObject;
   freeClayer();
   freeChannels();

#ifdef PV_USE_CUDA
   if (krUpdate) {
      delete krUpdate;
   }
   if (d_V) {
      delete d_V;
   }
   if (d_Activity) {
      delete d_Activity;
   }
   if (d_Datastore) {
      delete d_Datastore;
   }

#ifdef PV_USE_CUDNN
   if (cudnn_Datastore) {
      delete cudnn_Datastore;
   }
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   free(marginIndices);
   free(probes); // All probes are deleted by the HyPerCol, so probes[i] doesn't need to be deleted,
   // only the array itself.

   free(synchronizedMarginWidthLayers);

   free(triggerLayerName);
   free(triggerBehavior);
   free(triggerResetLayerName);
   free(initVTypeString);

   if (thread_gSyn) {
      for (int i = 0; i < parent->getNumThreads(); i++) {
         free(thread_gSyn[i]);
      }
      free(thread_gSyn);
   }
   delete publisher;
}

template <typename T>
int HyPerLayer::freeBuffer(T **buf) {
   free(*buf);
   *buf = NULL;
   return PV_SUCCESS;
}
// Declare the instantiations of allocateBuffer that occur in other .cpp files; otherwise you may
// get linker errors.
template int HyPerLayer::freeBuffer<float>(float **buf);
template int HyPerLayer::freeBuffer<int>(int **buf);

int HyPerLayer::freeRestrictedBuffer(float **buf) { return freeBuffer(buf); }

int HyPerLayer::freeExtendedBuffer(float **buf) { return freeBuffer(buf); }

int HyPerLayer::freeClayer() {
   pvcube_delete(clayer->activity);

   freeBuffer(&clayer->prevActivity);
   freeBuffer(&clayer->V);
   free(clayer);
   clayer = NULL;

   return PV_SUCCESS;
}

void HyPerLayer::freeChannels() {

#ifdef PV_USE_CUDA
   if (d_GSyn != NULL) {
      delete d_GSyn;
      d_GSyn = NULL;
   }
#ifdef PV_USE_CUDNN
   if (cudnn_GSyn != NULL) {
      delete cudnn_GSyn;
   }
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   // GSyn gets allocated in allocateDataStructures, but only if numChannels>0.
   if (GSyn) {
      assert(numChannels > 0);
      free(GSyn[0]); // conductances allocated contiguously so frees all buffer storage
      free(GSyn); // this frees the array pointers to separate conductance channels
      GSyn        = NULL;
      numChannels = 0;
   }
}

int HyPerLayer::allocateClayerBuffers() {
   // clayer fields numNeurons, numExtended, loc, xScale, yScale,
   // dx, dy, xOrigin, yOrigin were set in initClayer().
   assert(clayer);
   FatalIf(allocateV() != PV_SUCCESS, "%s: allocateV() failed.\n", getName());
   FatalIf(allocateActivity() != PV_SUCCESS, "%s: allocateActivity() failed.\n", getName());

   // athresher 11-4-16 TODO: Should these be called on non-spiking layers?
   FatalIf(allocatePrevActivity() != PV_SUCCESS, "%s: allocatePrevActivity() failed.\n", getName());
   for (int k = 0; k < getNumExtendedAllBatches(); k++) {
      clayer->prevActivity[k] = -10 * REFRACTORY_PERIOD; // allow neuron to fire at time t==0
   }
   return PV_SUCCESS;
}

template <typename T>
int HyPerLayer::allocateBuffer(T **buf, int bufsize, const char *bufname) {
   int status = PV_SUCCESS;
   *buf       = (T *)calloc(bufsize, sizeof(T));
   if (*buf == NULL) {
      ErrorLog().printf(
            "%s: rank %d process unable to allocate memory for %s: %s.\n",
            getDescription_c(),
            parent->columnId(),
            bufname,
            strerror(errno));
      status = PV_FAILURE;
   }
   return status;
}
// Declare the instantiations of allocateBuffer that occur in other .cpp files; otherwise you may
// get linker errors.
template int HyPerLayer::allocateBuffer<float>(float **buf, int bufsize, const char *bufname);
template int HyPerLayer::allocateBuffer<int>(int **buf, int bufsize, const char *bufname);

int HyPerLayer::allocateRestrictedBuffer(float **buf, char const *bufname) {
   return allocateBuffer(buf, getNumNeuronsAllBatches(), bufname);
}

int HyPerLayer::allocateExtendedBuffer(float **buf, char const *bufname) {
   return allocateBuffer(buf, getNumExtendedAllBatches(), bufname);
}

int HyPerLayer::allocateV() { return allocateRestrictedBuffer(&clayer->V, "membrane potential V"); }

int HyPerLayer::allocateActivity() {
   clayer->activity = pvcube_new(&clayer->loc, getNumExtendedAllBatches());
   return clayer->activity != NULL ? PV_SUCCESS : PV_FAILURE;
}

int HyPerLayer::allocatePrevActivity() {
   return allocateExtendedBuffer(&clayer->prevActivity, "time of previous activity");
}

int HyPerLayer::setLayerLoc(
      PVLayerLoc *layerLoc,
      float nxScale,
      float nyScale,
      int nf,
      int numBatches) {
   int status = PV_SUCCESS;

   Communicator *icComm = parent->getCommunicator();

   float nxglobalfloat = nxScale * parent->getNxGlobal();
   layerLoc->nxGlobal  = (int)nearbyintf(nxglobalfloat);
   if (std::fabs(nxglobalfloat - layerLoc->nxGlobal) > 0.0001f) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "nxScale of layer \"%s\" is incompatible with size of column.\n", getName());
         errorMessage.printf(
               "Column nx %d multiplied by nxScale %f must be an integer.\n",
               (double)parent->getNxGlobal(),
               (double)nxScale);
      }
      status = PV_FAILURE;
   }

   float nyglobalfloat = nyScale * parent->getNyGlobal();
   layerLoc->nyGlobal  = (int)nearbyintf(nyglobalfloat);
   if (std::fabs(nyglobalfloat - layerLoc->nyGlobal) > 0.0001f) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "nyScale of layer \"%s\" is incompatible with size of column.\n", getName());
         errorMessage.printf(
               "Column ny %d multiplied by nyScale %f must be an integer.\n",
               (double)parent->getNyGlobal(),
               (double)nyScale);
      }
      status = PV_FAILURE;
   }

   // partition input space based on the number of processor
   // columns and rows
   //

   if (layerLoc->nxGlobal % icComm->numCommColumns() != 0) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Size of HyPerLayer \"%s\" is not  compatible with the mpi configuration.\n", name);
         errorMessage.printf(
               "The layer has %d pixels horizontally, and there are %d mpi processes in a row, but "
               "%d does not divide %d.\n",
               layerLoc->nxGlobal,
               icComm->numCommColumns(),
               icComm->numCommColumns(),
               layerLoc->nxGlobal);
      }
      status = PV_FAILURE;
   }
   if (layerLoc->nyGlobal % icComm->numCommRows() != 0) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "Size of HyPerLayer \"%s\" is not  compatible with the mpi configuration.\n", name);
         errorMessage.printf(
               "The layer has %d pixels vertically, and there are %d mpi processes in a column, "
               "but %d does not divide %d.\n",
               layerLoc->nyGlobal,
               icComm->numCommRows(),
               icComm->numCommRows(),
               layerLoc->nyGlobal);
      }
      status = PV_FAILURE;
   }
   MPI_Barrier(icComm->communicator()); // If there is an error, make sure that MPI doesn't kill the
   // run before process 0 reports the error.
   if (status != PV_SUCCESS) {
      if (parent->columnId() == 0) {
         ErrorLog().printf("setLayerLoc failed for %s.\n", getDescription_c());
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

   layerLoc->kb0          = parent->commBatch() * numBatches;
   layerLoc->nbatchGlobal = parent->numCommBatches() * numBatches;

   // halo is set in calls to updateClayerMargin
   layerLoc->halo.lt = 0;
   layerLoc->halo.rt = 0;
   layerLoc->halo.dn = 0;
   layerLoc->halo.up = 0;

   return 0;
}

void HyPerLayer::calcNumExtended() {
   PVLayerLoc const *loc = getLayerLoc();
   clayer->numExtended   = (loc->nx + loc->halo.lt + loc->halo.rt)
                         * (loc->ny + loc->halo.dn + loc->halo.up) * loc->nf;
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
   GSyn       = NULL;
   if (numChannels > 0) {
      GSyn = (float **)malloc(numChannels * sizeof(float *));
      if (GSyn == NULL) {
         status = PV_FAILURE;
         return status;
      }

      GSyn[0] = (float *)calloc(getNumNeuronsAllBatches() * numChannels, sizeof(float));
      // All channels allocated at once and contiguously.  resetGSynBuffers_HyPerLayer() assumes
      // this is true, to make it easier to port to GPU.
      if (GSyn[0] == NULL) {
         status = PV_FAILURE;
         return status;
      }

      for (int m = 1; m < numChannels; m++) {
         GSyn[m] = GSyn[0] + m * getNumNeuronsAllBatches();
      }
   }

   return status;
}

void HyPerLayer::addPublisher() {
   MPIBlock const *mpiBlock = parent->getCommunicator()->getLocalMPIBlock();
   publisher = new Publisher(*mpiBlock, clayer->activity, getNumDelayLevels(), getSparseFlag());
}

void HyPerLayer::checkpointPvpActivityFloat(
      Checkpointer *checkpointer,
      char const *bufferName,
      float *pvpBuffer,
      bool extended) {
   bool registerSucceeded = checkpointer->registerCheckpointEntry(
         std::make_shared<CheckpointEntryPvpBuffer<float>>(
               getName(),
               bufferName,
               checkpointer->getMPIBlock(),
               pvpBuffer,
               getLayerLoc(),
               extended),
         false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         getDescription_c(),
         bufferName);
}

void HyPerLayer::checkpointRandState(
      Checkpointer *checkpointer,
      char const *bufferName,
      Random *randState,
      bool extendedFlag) {
   bool registerSucceeded = checkpointer->registerCheckpointEntry(
         std::make_shared<CheckpointEntryRandState>(
               getName(),
               bufferName,
               checkpointer->getMPIBlock(),
               randState->getRNG(0),
               getLayerLoc(),
               extendedFlag),
         false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         getDescription_c(),
         bufferName);
}

int HyPerLayer::initializeState() {
   int status = setInitialValues();
   return status;
}

#ifdef PV_USE_CUDA
int HyPerLayer::copyInitialStateToGPU() {
   if (mUpdateGpu) {
      float *h_V = getV();
      if (h_V != NULL) {
         PVCuda::CudaBuffer *d_V = getDeviceV();
         assert(d_V);
         d_V->copyToDevice(h_V);
      }

      PVCuda::CudaBuffer *d_activity = getDeviceActivity();
      assert(d_activity);
      float *h_activity = getCLayer()->activity->data;
      d_activity->copyToDevice(h_activity);
   }
   return PV_SUCCESS;
}

#endif // PV_USE_CUDA

int HyPerLayer::setInitialValues() {
   int status = PV_SUCCESS;
   status     = initializeV();
   if (status == PV_SUCCESS)
      initializeActivity();
   return status;
}

int HyPerLayer::initializeV() {
   int status = PV_SUCCESS;
   if (getV() != nullptr && mInitVObject != nullptr) {
      status = mInitVObject->calcV(getV(), getLayerLoc());
   }
   return status;
}

int HyPerLayer::initializeActivity() {
   int status = setActivity();
   return status;
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
   this->parent->parameters()->ioParamString(
         ioFlag, this->getName(), "dataType", &dataTypeString, NULL, false /*warnIfAbsent*/);
   if (dataTypeString == NULL) {
      // Default value
      dataType = PV_FLOAT;
      return;
   }
   if (!strcmp(dataTypeString, "float")) {
      dataType = PV_FLOAT;
   }
   else if (!strcmp(dataTypeString, "int")) {
      dataType = PV_INT;
   }
   else {
      Fatal() << "BaseLayer \"" << name
              << "\": dataType not recognized, can be \"float\" or \"int\"\n";
   }
}

void HyPerLayer::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   parent->parameters()->ioParamValue(
         ioFlag, name, "updateGpu", &mUpdateGpu, mUpdateGpu, true /*warnIfAbsent*/);
   mUsingGPUFlag = mUpdateGpu;
#else // PV_USE_CUDA
   bool mUpdateGpu = false;
   parent->parameters()->ioParamValue(
         ioFlag, name, "updateGpu", &mUpdateGpu, mUpdateGpu, false /*warnIfAbsent*/);
   if (parent->columnId() == 0) {
      FatalIf(
            mUpdateGpu,
            "%s: updateGpu is set to true, but PetaVision was compiled without GPU acceleration.\n",
            getDescription_c());
   }
#endif // PV_USE_CUDA
}

void HyPerLayer::ioParam_nxScale(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nxScale", &nxScale, nxScale);
}

void HyPerLayer::ioParam_nyScale(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nyScale", &nyScale, nyScale);
}

void HyPerLayer::ioParam_nf(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "nf", &numFeatures, numFeatures);
}

void HyPerLayer::ioParam_phase(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "phase", &phase, phase);
   if (ioFlag == PARAMS_IO_READ && phase < 0) {
      if (parent->columnId() == 0)
         Fatal().printf(
               "%s: phase must be >= 0 (given value was %d).\n", getDescription_c(), phase);
   }
}

void HyPerLayer::ioParam_mirrorBCflag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "mirrorBCflag", &mirrorBCflag, mirrorBCflag);
}

void HyPerLayer::ioParam_valueBC(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "mirrorBCflag"));
   if (!mirrorBCflag) {
      parent->parameters()->ioParamValue(ioFlag, name, "valueBC", &valueBC, (float)0);
   }
}

void HyPerLayer::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "initializeFromCheckpointFlag",
         &initializeFromCheckpointFlag,
         initializeFromCheckpointFlag,
         true /*warnIfAbsent*/);
}

void HyPerLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "InitVType", &initVTypeString, "ConstantV", true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      BaseObject *object = Factory::instance()->createByKeyword(initVTypeString, name, parent);
      mInitVObject       = dynamic_cast<BaseInitV *>(object);
      if (mInitVObject == nullptr) {
         ErrorLog().printf("%s: unable to create InitV object\n", getDescription_c());
         abort();
      }
   }
   if (mInitVObject != nullptr) {
      mInitVObject->ioParamsFillGroup(ioFlag);
   }
}

void HyPerLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(
         ioFlag, name, "triggerLayerName", &triggerLayerName, NULL, false /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      if (triggerLayerName && !strcmp(name, triggerLayerName)) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: triggerLayerName cannot be the same as the name of the layer itself.\n",
                  getDescription_c());
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      triggerFlag = (triggerLayerName != NULL && triggerLayerName[0] != '\0');
   }
}

// triggerFlag was deprecated Aug 7, 2015.
// Setting triggerLayerName to a nonempty string has the effect of triggerFlag=true, and
// setting triggerLayerName to NULL or "" has the effect of triggerFlag=false.
// While triggerFlag is being deprecated, it is an error for triggerFlag to be false
// and triggerLayerName to be a nonempty string.
void HyPerLayer::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (ioFlag == PARAMS_IO_READ && parent->parameters()->present(name, "triggerFlag")) {
      bool flagFromParams = false;
      parent->parameters()->ioParamValue(
            ioFlag, name, "triggerFlag", &flagFromParams, flagFromParams);
      if (parent->columnId() == 0) {
         WarnLog(triggerFlagMessage);
         triggerFlagMessage.printf("%s: triggerFlag has been deprecated.\n", getDescription_c());
         triggerFlagMessage.printf(
               "   If triggerLayerName is a nonempty string, triggering will be on;\n");
         triggerFlagMessage.printf(
               "   if triggerLayerName is empty or null, triggering will be off.\n");
         if (parent->columnId() == 0) {
            if (flagFromParams != triggerFlag) {
               ErrorLog(errorMessage);
               errorMessage.printf("triggerLayerName=", name);
               if (triggerLayerName) {
                  errorMessage.printf("\"%s\"", triggerLayerName);
               }
               else {
                  errorMessage.printf("NULL");
               }
               errorMessage.printf(
                     " implies triggerFlag=%s but triggerFlag was set in params to %s\n",
                     triggerFlag ? "true" : "false",
                     flagFromParams ? "true" : "false");
            }
         }
      }
      if (flagFromParams != triggerFlag) {
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
}

void HyPerLayer::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      parent->parameters()->ioParamValue(
            ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
      if (triggerOffset < 0) {
         if (parent->columnId() == 0) {
            Fatal().printf(
                  "%s: TriggerOffset (%f) must be positive\n", getDescription_c(), triggerOffset);
         }
      }
   }
}
void HyPerLayer::ioParam_triggerBehavior(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      parent->parameters()->ioParamString(
            ioFlag,
            name,
            "triggerBehavior",
            &triggerBehavior,
            "updateOnlyOnTrigger",
            true /*warnIfAbsent*/);
      if (triggerBehavior == NULL || !strcmp(triggerBehavior, "")) {
         free(triggerBehavior);
         triggerBehavior     = strdup("updateOnlyOnTrigger");
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
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: triggerBehavior=\"%s\" is unrecognized.\n",
                  getDescription_c(),
                  triggerBehavior);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   else {
      triggerBehaviorType = NO_TRIGGER;
   }
}

void HyPerLayer::ioParam_triggerResetLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "triggerBehavior"));
      if (!strcmp(triggerBehavior, "resetStateOnTrigger")) {
         parent->parameters()->ioParamStringRequired(
               ioFlag, name, "triggerResetLayerName", &triggerResetLayerName);
      }
   }
}

void HyPerLayer::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "writeStep", &writeStep, parent->getDeltaTime());
}

void HyPerLayer::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep >= 0.0) {
      double start_time = parent->getStartTime();
      parent->parameters()->ioParamValue(
            ioFlag, name, "initialWriteTime", &initialWriteTime, start_time);
      if (ioFlag == PARAMS_IO_READ && writeStep > 0.0 && initialWriteTime < start_time) {
         double storeInitialWriteTime = initialWriteTime;
         while (initialWriteTime < start_time) {
            initialWriteTime += writeStep;
         }
         if (parent->columnId() == 0) {
            WarnLog(warningMessage);
            warningMessage.printf(
                  "%s: initialWriteTime %f is earlier than start time %f.  Adjusting "
                  "initialWriteTime:\n",
                  getDescription_c(),
                  initialWriteTime,
                  start_time);
            warningMessage.printf("    initialWriteTime adjusted to %f\n", initialWriteTime);
         }
      }
   }
}

void HyPerLayer::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ && !parent->parameters()->present(name, "sparseLayer")
       && parent->parameters()->present(name, "writeSparseActivity")) {
      Fatal().printf("writeSparseActivity is obsolete. Use sparseLayer instead.\n");
   }
   // writeSparseActivity was deprecated Nov 4, 2014 and marked obsolete Mar 14, 2017.
   parent->parameters()->ioParamValue(ioFlag, name, "sparseLayer", &sparseLayer, false);
}

// writeSparseValues is obsolete as of Mar 14, 2017.
void HyPerLayer::ioParam_writeSparseValues(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "sparseLayer"));
      if (sparseLayer && parent->parameters()->present(name, "writeSparseValues")) {
         WarnLog() << "writeSparseValues parameter, defined in " << getDescription()
                   << ", is obsolete.\n";
         bool writeSparseValues;
         parent->parameters()->ioParamValue(
               ioFlag, name, "writeSparseValues", &writeSparseValues, true /*default value*/);
         if (!writeSparseValues) {
            WarnLog() << "The sparse-values format is used for all sparse layers.\n";
         }
      }
   }
}

int HyPerLayer::respond(std::shared_ptr<BaseMessage const> message) {
   int status = BaseLayer::respond(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   else if (auto castMessage = std::dynamic_pointer_cast<LayerSetMaxPhaseMessage const>(message)) {
      return respondLayerSetMaxPhase(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<LayerWriteParamsMessage const>(message)) {
      return respondLayerWriteParams(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<LayerProbeWriteParamsMessage const>(message)) {
      return respondLayerProbeWriteParams(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<LayerClearProgressFlagsMessage const>(message)) {
      return respondLayerClearProgressFlags(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<LayerUpdateStateMessage const>(message)) {
      return respondLayerUpdateState(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<LayerRecvSynapticInputMessage const>(message)) {
      return respondLayerRecvSynapticInput(castMessage);
   }
#ifdef PV_USE_CUDA
   else if (auto castMessage = std::dynamic_pointer_cast<LayerCopyFromGpuMessage const>(message)) {
      return respondLayerCopyFromGpu(castMessage);
   }
#endif // PV_USE_CUDA
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<LayerAdvanceDataStoreMessage const>(message)) {
      return respondLayerAdvanceDataStore(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<LayerPublishMessage const>(message)) {
      return respondLayerPublish(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(message)) {
      return respondLayerOutputState(castMessage);
   }
   else if (
         auto castMessage = std::dynamic_pointer_cast<LayerCheckNotANumberMessage const>(message)) {
      return respondLayerCheckNotANumber(castMessage);
   }
   else {
      return status;
   }
}

int HyPerLayer::respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message) {
   return setMaxPhase(message->mMaxPhase);
}

int HyPerLayer::respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message) {
   return writeParams();
}

int HyPerLayer::respondLayerProbeWriteParams(
      std::shared_ptr<LayerProbeWriteParamsMessage const> message) {
   return outputProbeParams();
}

int HyPerLayer::respondLayerClearProgressFlags(
      std::shared_ptr<LayerClearProgressFlagsMessage const> message) {
   return clearProgressFlags();
}

int HyPerLayer::respondLayerRecvSynapticInput(
      std::shared_ptr<LayerRecvSynapticInputMessage const> message) {
   int status = PV_SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
#ifdef PV_USE_CUDA
   if (message->mRecvOnGpuFlag != mRecvGpu) {
      return status;
   }
#endif // PV_USE_CUDA
   if (mHasReceived) {
      return status;
   }
   if (*(message->mSomeLayerHasActed) or !isAllInputReady()) {
      *(message->mSomeLayerIsPending) = true;
      return status;
   }
   resetGSynBuffers(message->mTime, message->mDeltaT); // deltaTimeAdapt is not used

   message->mTimer->start();
   recvAllSynapticInput();
   mHasReceived                   = true;
   *(message->mSomeLayerHasActed) = true;
   message->mTimer->stop();

   return status;
}

int HyPerLayer::respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message) {
   int status = PV_SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
#ifdef PV_USE_CUDA
   if (message->mRecvOnGpuFlag != mRecvGpu) {
      return status;
   }
   if (message->mUpdateOnGpuFlag != mUpdateGpu) {
      return status;
   }
#endif // PV_USE_CUDA
   if (mHasUpdated) {
      return status;
   }
   if (*(message->mSomeLayerHasActed) or !mHasReceived) {
      *(message->mSomeLayerIsPending) = true;
      return status;
   }
   status                         = callUpdateState(message->mTime, message->mDeltaT);
   mHasUpdated                    = true;
   *(message->mSomeLayerHasActed) = true;
   return status;
}

#ifdef PV_USE_CUDA
int HyPerLayer::respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   int status = PV_SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   message->mTimer->start();
   copyAllActivityFromDevice();
   copyAllVFromDevice();
   copyAllGSynFromDevice();
   addGpuTimers();
   message->mTimer->stop();
   return status;
}
#endif // PV_USE_CUDA

int HyPerLayer::respondLayerAdvanceDataStore(
      std::shared_ptr<LayerAdvanceDataStoreMessage const> message) {
   if (message->mPhase < 0 || message->mPhase == getPhase()) {
      publisher->increaseTimeLevel();
   }
   return PV_SUCCESS;
}

int HyPerLayer::respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message) {
   int status = PV_SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   publish(parent->getCommunicator(), message->mTime);
   return status;
}

int HyPerLayer::respondLayerCheckNotANumber(
      std::shared_ptr<LayerCheckNotANumberMessage const> message) {
   int status = PV_SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   auto layerData = getLayerData();
   int const N    = getNumExtendedAllBatches();
   for (int n = 0; n < N; n++) {
      float a = layerData[n];
      if (a != a) {
         status = PV_FAILURE;
         break;
      }
   }
   if (status != PV_SUCCESS) {
      if (parent->columnId() == 0) {
         ErrorLog() << getDescription()
                    << " has not-a-number values in the activity buffer.  Exiting.\n";
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

int HyPerLayer::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   int status = PV_SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   status = outputState(message->mTime); // also calls layer probes' outputState
   return status;
}

int HyPerLayer::clearProgressFlags() {
   mHasReceived = false;
   mHasUpdated  = false;
   return PV_SUCCESS;
}

#ifdef PV_USE_CUDA

int HyPerLayer::allocateUpdateKernel() {
   Fatal() << "Layer \"" << name << "\" of type " << getKeyword()
           << " does not support updating on gpus yet\n";
   return -1;
}

/**
 * Allocate GPU buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int HyPerLayer::allocateDeviceBuffers() {
   int status = 0;

   const size_t size    = getNumNeuronsAllBatches() * sizeof(float);
   const size_t size_ex = getNumExtendedAllBatches() * sizeof(float);

   PVCuda::CudaDevice *device = parent->getDevice();

   // Allocate based on which flags are set
   if (allocDeviceV) {
      d_V = device->createBuffer(size, &description);
   }

   if (allocDeviceDatastore) {
      d_Datastore = device->createBuffer(size_ex, &description);
      assert(d_Datastore);
#ifdef PV_USE_CUDNN
      cudnn_Datastore = device->createBuffer(size_ex, &description);
      assert(cudnn_Datastore);
#endif
   }

   if (allocDeviceActiveIndices) {
      d_numActive     = device->createBuffer(parent->getNBatch() * sizeof(long), &description);
      d_ActiveIndices = device->createBuffer(
            getNumExtendedAllBatches() * sizeof(SparseList<float>::Entry), &description);
      assert(d_ActiveIndices);
   }

   if (allocDeviceActivity) {
      d_Activity = device->createBuffer(size_ex, &description);
   }

   // d_GSyn is the entire gsyn buffer. cudnn_GSyn is only one gsyn channel
   if (allocDeviceGSyn) {
      d_GSyn = device->createBuffer(size * numChannels, &description);
      assert(d_GSyn);
#ifdef PV_USE_CUDNN
      cudnn_GSyn = device->createBuffer(size, &description);
#endif
   }

   return status;
}

#endif // PV_USE_CUDA

int HyPerLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
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
   if (triggerFlag) {
      triggerLayer = message->lookup<HyPerLayer>(std::string(triggerLayerName));
      if (triggerLayer == NULL) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: triggerLayerName \"%s\" is not a layer in the HyPerCol.\n",
                  getDescription_c(),
                  triggerLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (triggerBehaviorType == RESETSTATE_TRIGGER) {
         char const *resetLayerName = NULL; // Will point to name of actual resetLayer, whether
         // triggerResetLayerName is blank (in which case
         // resetLayerName==triggerLayerName) or not
         if (triggerResetLayerName == NULL || triggerResetLayerName[0] == '\0') {
            resetLayerName    = triggerLayerName;
            triggerResetLayer = triggerLayer;
         }
         else {
            resetLayerName    = triggerResetLayerName;
            triggerResetLayer = message->lookup<HyPerLayer>(std::string(triggerResetLayerName));
            if (triggerResetLayer == NULL) {
               if (parent->columnId() == 0) {
                  ErrorLog().printf(
                        "%s: triggerResetLayerName \"%s\" is not a layer in the HyPerCol.\n",
                        getDescription_c(),
                        triggerResetLayerName);
               }
               MPI_Barrier(parent->getCommunicator()->communicator());
               exit(EXIT_FAILURE);
            }
         }
         // Check that triggerResetLayer and this layer have the same (restricted) dimensions.
         // Do we need to postpone until triggerResetLayer has finished its communicateInitInfo?
         PVLayerLoc const *triggerLoc = triggerResetLayer->getLayerLoc();
         PVLayerLoc const *localLoc   = this->getLayerLoc();
         if (triggerLoc->nxGlobal != localLoc->nxGlobal
             || triggerLoc->nyGlobal != localLoc->nyGlobal
             || triggerLoc->nf != localLoc->nf) {
            if (parent->columnId() == 0) {
               Fatal(errorMessage);
               errorMessage.printf(
                     "%s: triggerResetLayer \"%s\" has incompatible dimensions.\n",
                     getDescription_c(),
                     resetLayerName);
               errorMessage.printf(
                     "    \"%s\" is %d-by-%d-by-%d and \"%s\" is %d-by-%d-by-%d.\n",
                     name,
                     localLoc->nxGlobal,
                     localLoc->nyGlobal,
                     localLoc->nf,
                     resetLayerName,
                     triggerLoc->nxGlobal,
                     triggerLoc->nyGlobal,
                     triggerLoc->nf);
            }
         }
      }
   }

#ifdef PV_USE_CUDA
   // Here, the connection tells all participating recev layers to allocate memory on gpu
   // if receive from gpu is set. These buffers should be set in allocate
   if (mUpdateGpu) {
      this->setAllocDeviceGSyn();
      this->setAllocDeviceV();
      this->setAllocDeviceActivity();
   }
#endif

   int status = PV_SUCCESS;

   return status;
}

int HyPerLayer::setMaxPhase(int *maxPhase) {
   if (*maxPhase < phase) {
      *maxPhase = phase;
   }
   return PV_SUCCESS;
}

void HyPerLayer::addRecvConn(BaseConnection *conn) {
   FatalIf(
         conn->postSynapticLayer() != this,
         "%s called addRecvConn for %s, but \"%s\" is not the post-synaptic layer for \"%s\"\n.",
         conn->getDescription_c(),
         getDescription_c(),
         getName(),
         conn->getName());
#ifdef PV_USE_CUDA
   // CPU connections must run first to avoid race conditions
   if (!conn->getReceiveGpu()) {
      recvConns.insert(recvConns.begin(), conn);
   }
   // Otherwise, add to the back. If no gpus at all, just add to back
   else
#endif
   {
      recvConns.push_back(conn);
#ifdef PV_USE_CUDA
      // If it is receiving from gpu, set layer flag as such
      mRecvGpu = true;
#endif
   }
}

int HyPerLayer::openOutputStateFile(Checkpointer *checkpointer) {
   pvAssert(writeStep >= 0);

   if (checkpointer->getMPIBlock()->getRank() == 0) {
      std::string outputStatePath(getName());
      outputStatePath.append(".pvp");

      std::string checkpointLabel(getName());
      checkpointLabel.append("_filepos");

      bool createFlag    = checkpointer->getCheckpointReadDirectory().empty();
      mOutputStateStream = new CheckpointableFileStream(
            outputStatePath.c_str(), createFlag, checkpointer, checkpointLabel);
   }
   return PV_SUCCESS;
}

void HyPerLayer::synchronizeMarginWidth(HyPerLayer *layer) {
   if (layer == this) {
      return;
   }
   assert(layer->getLayerLoc() != NULL && this->getLayerLoc() != NULL);
   HyPerLayer **newSynchronizedMarginWidthLayers =
         (HyPerLayer **)calloc(numSynchronizedMarginWidthLayers + 1, sizeof(HyPerLayer *));
   assert(newSynchronizedMarginWidthLayers);
   if (numSynchronizedMarginWidthLayers > 0) {
      for (int k = 0; k < numSynchronizedMarginWidthLayers; k++) {
         newSynchronizedMarginWidthLayers[k] = synchronizedMarginWidthLayers[k];
      }
      free(synchronizedMarginWidthLayers);
   }
   else {
      assert(synchronizedMarginWidthLayers == NULL);
   }
   synchronizedMarginWidthLayers = newSynchronizedMarginWidthLayers;
   synchronizedMarginWidthLayers[numSynchronizedMarginWidthLayers] = layer;
   numSynchronizedMarginWidthLayers++;

   equalizeMargins(this, layer);

   return;
}

int HyPerLayer::equalizeMargins(HyPerLayer *layer1, HyPerLayer *layer2) {
   int border1, border2, maxborder, result;
   int status = PV_SUCCESS;

   border1   = layer1->getLayerLoc()->halo.lt;
   border2   = layer2->getLayerLoc()->halo.lt;
   maxborder = border1 > border2 ? border1 : border2;
   layer1->requireMarginWidth(maxborder, &result, 'x');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   layer2->requireMarginWidth(maxborder, &result, 'x');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "Error in rank %d process: unable to synchronize x-margin widths of layers \"%s\" and "
            "\"%s\" to %d\n",
            layer1->parent->columnId(),
            layer1->getName(),
            layer2->getName(),
            maxborder);
      ;
   }
   assert(
         layer1->getLayerLoc()->halo.lt == layer2->getLayerLoc()->halo.lt
         && layer1->getLayerLoc()->halo.rt == layer2->getLayerLoc()->halo.rt
         && layer1->getLayerLoc()->halo.lt == layer1->getLayerLoc()->halo.rt
         && layer1->getLayerLoc()->halo.lt == maxborder);

   border1   = layer1->getLayerLoc()->halo.dn;
   border2   = layer2->getLayerLoc()->halo.dn;
   maxborder = border1 > border2 ? border1 : border2;
   layer1->requireMarginWidth(maxborder, &result, 'y');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   layer2->requireMarginWidth(maxborder, &result, 'y');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "Error in rank %d process: unable to synchronize y-margin widths of layers \"%s\" and "
            "\"%s\" to %d\n",
            layer1->parent->columnId(),
            layer1->getName(),
            layer2->getName(),
            maxborder);
      ;
   }
   assert(
         layer1->getLayerLoc()->halo.dn == layer2->getLayerLoc()->halo.dn
         && layer1->getLayerLoc()->halo.up == layer2->getLayerLoc()->halo.up
         && layer1->getLayerLoc()->halo.dn == layer1->getLayerLoc()->halo.up
         && layer1->getLayerLoc()->halo.dn == maxborder);
   return status;
}

int HyPerLayer::allocateDataStructures() {
   // Once initialize and communicateInitInfo have been called, HyPerLayer has the
   // information it needs to allocate the membrane potential buffer V, the
   // activity buffer activity->data, and the data store.
   int status = PV_SUCCESS;

   // Doing this check here, since trigger layers are being set up in communicateInitInfo
   // If the magnitude of the trigger offset is bigger than the delta update time, then error
   if (triggerFlag) {
      double deltaUpdateTime = getDeltaUpdateTime();
      if (deltaUpdateTime != -1 && triggerOffset >= deltaUpdateTime) {
         Fatal().printf(
               "%s error in rank %d process: TriggerOffset (%f) must be lower than the change in "
               "update time (%f) \n",
               getDescription_c(),
               parent->columnId(),
               triggerOffset,
               deltaUpdateTime);
      }
   }

   allocateClayerBuffers();

   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   PVHalo const *halo    = &loc->halo;

   // If not mirroring, fill the boundaries with the value in the valueBC param
   if (!useMirrorBCs() && getValueBC() != 0.0f) {
      int idx = 0;
      for (int batch = 0; batch < loc->nbatch; batch++) {
         for (int b = 0; b < halo->up; b++) {
            for (int k = 0; k < (nx + halo->lt + halo->rt) * nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
         }
         for (int y = 0; y < ny; y++) {
            for (int k = 0; k < halo->lt * nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
            idx += nx * nf;
            for (int k = 0; k < halo->rt * nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
         }
         for (int b = 0; b < halo->dn; b++) {
            for (int k = 0; k < (nx + halo->lt + halo->rt) * nf; k++) {
               clayer->activity->data[idx] = getValueBC();
               idx++;
            }
         }
      }
      assert(idx == getNumExtendedAllBatches());
   }

   // allocate storage for the input conductance arrays
   status = allocateBuffers();
   assert(status == PV_SUCCESS);

   // Allocate temp buffers if needed, 1 for each thread
   if (parent->getNumThreads() > 1) {
      thread_gSyn = (float **)malloc(sizeof(float *) * parent->getNumThreads());
      assert(thread_gSyn);

      // Assign thread_gSyn to different points of tempMem
      for (int i = 0; i < parent->getNumThreads(); i++) {
         float *tempMem = (float *)malloc(sizeof(float) * getNumNeuronsAllBatches());
         if (!tempMem) {
            Fatal().printf(
                  "HyPerLayer \"%s\" error: rank %d unable to allocate %zu memory for thread_gSyn: "
                  "%s\n",
                  name,
                  parent->columnId(),
                  sizeof(float) * getNumNeuronsAllBatches(),
                  strerror(errno));
         }
         thread_gSyn[i] = tempMem;
      }
   }

// Allocate cuda stuff on gpu if set
#ifdef PV_USE_CUDA
   status = allocateDeviceBuffers();
   // Allocate receive from post kernel
   if (status == 0) {
      status = PV_SUCCESS;
   }
   else {
      Fatal().printf(
            "%s unable to allocate device memory in rank %d process: %s\n",
            getDescription_c(),
            parent->columnId(),
            strerror(errno));
   }
   if (mUpdateGpu) {
      // This function needs to be overwritten as needed on a subclass basis
      status = allocateUpdateKernel();
      if (status == 0) {
         status = PV_SUCCESS;
      }
   }
#endif

   addPublisher();

   return status;
}

/*
 * Call this routine to increase the number of levels in the data store ring buffer.
 * Calls to this routine after the data store has been initialized will have no effect.
 * The routine returns the new value of numDelayLevels
 */
int HyPerLayer::increaseDelayLevels(int neededDelay) {
   if (numDelayLevels < neededDelay + 1)
      numDelayLevels = neededDelay + 1;
   if (numDelayLevels > MAX_F_DELAY)
      numDelayLevels = MAX_F_DELAY;
   return numDelayLevels;
}

int HyPerLayer::requireMarginWidth(int marginWidthNeeded, int *marginWidthResult, char axis) {
   // TODO: Is there a good way to handle x- and y-axis margins without so much duplication of code?
   // Navigating through the halo makes it difficult to combine cases.
   PVLayerLoc *loc = &clayer->loc;
   PVHalo *halo    = &loc->halo;
   switch (axis) {
      case 'x':
         *marginWidthResult = xmargin;
         if (xmargin < marginWidthNeeded) {
            assert(clayer);
            if (parent->columnId() == 0) {
               InfoLog().printf(
                     "%s: adjusting x-margin width from %d to %d\n",
                     getDescription_c(),
                     xmargin,
                     marginWidthNeeded);
            }
            xmargin  = marginWidthNeeded;
            halo->lt = xmargin;
            halo->rt = xmargin;
            calcNumExtended();
            assert(axis == 'x' && getLayerLoc()->halo.lt == getLayerLoc()->halo.rt);
            *marginWidthResult = xmargin;
            if (synchronizedMarginWidthLayers != NULL) {
               for (int k = 0; k < numSynchronizedMarginWidthLayers; k++) {
                  HyPerLayer *l = synchronizedMarginWidthLayers[k];
                  if (l->getLayerLoc()->halo.lt < marginWidthNeeded) {
                     synchronizedMarginWidthLayers[k]->requireMarginWidth(
                           marginWidthNeeded, marginWidthResult, axis);
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
            if (parent->columnId() == 0) {
               InfoLog().printf(
                     "%s: adjusting y-margin width from %d to %d\n",
                     getDescription_c(),
                     ymargin,
                     marginWidthNeeded);
            }
            ymargin  = marginWidthNeeded;
            halo->dn = ymargin;
            halo->up = ymargin;
            calcNumExtended();
            assert(axis == 'y' && getLayerLoc()->halo.dn == getLayerLoc()->halo.up);
            *marginWidthResult = ymargin;
            if (synchronizedMarginWidthLayers != NULL) {
               for (int k = 0; k < numSynchronizedMarginWidthLayers; k++) {
                  HyPerLayer *l = synchronizedMarginWidthLayers[k];
                  if (l->getLayerLoc()->halo.up < marginWidthNeeded) {
                     synchronizedMarginWidthLayers[k]->requireMarginWidth(
                           marginWidthNeeded, marginWidthResult, axis);
                  }
                  assert(l->getLayerLoc()->halo.dn == getLayerLoc()->halo.dn);
                  assert(l->getLayerLoc()->halo.up == getLayerLoc()->halo.up);
               }
            }
         }
         break;
      default: assert(0); break;
   }
   return PV_SUCCESS;
}

int HyPerLayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   if (channelNeeded >= numChannels) {
      int numOldChannels = numChannels;
      numChannels        = channelNeeded + 1;
   }
   *numChannelsResult = numChannels;

   return PV_SUCCESS;
}

/**
 * Returns the activity data for the layer.  This data is in the
 * extended space (with margins).
 */
const float *HyPerLayer::getLayerData(int delay) {
   PVLayerCube cube = publisher->createCube(delay);
   return cube.data;
}

int HyPerLayer::mirrorInteriorToBorder(PVLayerCube *cube, PVLayerCube *border) {
   assert(cube->numItems == border->numItems);
   assert(localDimensionsEqual(&cube->loc, &border->loc));

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

int HyPerLayer::registerData(Checkpointer *checkpointer) {
   int status = BaseLayer::registerData(checkpointer);
   checkpointPvpActivityFloat(checkpointer, "A", getActivity(), true /*extended*/);
   if (getV() != nullptr) {
      checkpointPvpActivityFloat(checkpointer, "V", getV(), false /*not extended*/);
   }
   publisher->checkpointDataStore(checkpointer, getName(), "Delays");
   checkpointer->registerCheckpointData(
         std::string(getName()),
         std::string("lastUpdateTime"),
         &mLastUpdateTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   checkpointer->registerCheckpointData(
         std::string(getName()),
         std::string("nextWrite"),
         &writeTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);

   if (writeStep >= 0.0) {
      openOutputStateFile(checkpointer);
      if (sparseLayer) {
         checkpointer->registerCheckpointData(
               std::string(getName()),
               std::string("numframes_sparse"),
               &writeActivitySparseCalls,
               (std::size_t)1,
               true /*broadcast*/,
               false /*not constant*/);
      }
      else {
         checkpointer->registerCheckpointData(
               std::string(getName()),
               std::string("numframes"),
               &writeActivityCalls,
               (std::size_t)1,
               true /*broadcast*/,
               false /*not constant*/);
      }
   }

   // Timers

   update_timer = new Timer(getName(), "layer", "update ");
   checkpointer->registerTimer(update_timer);

   recvsyn_timer = new Timer(getName(), "layer", "recvsyn");
   checkpointer->registerTimer(recvsyn_timer);
#ifdef PV_USE_CUDA
   auto cudaDevice = parent->getDevice();
   if (cudaDevice) {
      gpu_update_timer = new PVCuda::CudaTimer(getName(), "layer", "gpuupdate");
      gpu_update_timer->setStream(cudaDevice->getStream());
      checkpointer->registerTimer(gpu_update_timer);

      gpu_recvsyn_timer = new PVCuda::CudaTimer(getName(), "layer", "gpurecvsyn");
      gpu_recvsyn_timer->setStream(cudaDevice->getStream());
      checkpointer->registerTimer(gpu_recvsyn_timer);
   }
#endif // PV_USE_CUDA

   publish_timer = new Timer(getName(), "layer", "publish");
   checkpointer->registerTimer(publish_timer);

   timescale_timer = new Timer(getName(), "layer", "timescale");
   checkpointer->registerTimer(timescale_timer);

   io_timer = new Timer(getName(), "layer", "io     ");
   checkpointer->registerTimer(io_timer);

   if (mInitVObject) {
      auto message = std::make_shared<RegisterDataMessage<Checkpointer>>(checkpointer);
      mInitVObject->respond(message);
   }

   return PV_SUCCESS;
}

double HyPerLayer::getDeltaUpdateTime() {
   if (triggerLayer != NULL && triggerBehaviorType == UPDATEONLY_TRIGGER) {
      return getDeltaTriggerTime();
   }
   else {
      return parent->getDeltaTime();
   }
}

double HyPerLayer::getDeltaTriggerTime() {
   if (triggerLayer != NULL) {
      return triggerLayer->getDeltaUpdateTime();
   }
   else {
      return -1;
   }
}

bool HyPerLayer::needUpdate(double simTime, double dt) {
   if (getDeltaUpdateTime() <= 0) {
      return false;
   }
   if (mLastUpdateTime == simTime + triggerOffset) {
      return true;
   }
   double timeToCheck = mLastUpdateTime;
   if (triggerLayer != nullptr && triggerBehaviorType == UPDATEONLY_TRIGGER) {
      timeToCheck = triggerLayer->getLastUpdateTime();

      // If our target layer updates this tick, so do we
      if (timeToCheck == simTime && triggerOffset == 0) {
         return true;
      }
   }
   if (simTime + triggerOffset >= timeToCheck + getDeltaUpdateTime()
       && simTime + triggerOffset + dt <= timeToCheck + getDeltaUpdateTime() + dt) {
      return true;
   }
   return false;
}

bool HyPerLayer::needReset(double simTime, double dt) {
   if (triggerLayer == nullptr) {
      return false;
   }
   if (triggerBehaviorType != RESETSTATE_TRIGGER) {
      return false;
   }
   if (getDeltaTriggerTime() <= 0) {
      return false;
   }
   if (simTime >= mLastTriggerTime + getDeltaTriggerTime()) {
      // TODO: test "simTime > mLastTriggerTime + getDeltaTriggerTime() - 0.5 * dt",
      // to avoid roundoff issues.
      return true;
   }
   return false;
}

int HyPerLayer::callUpdateState(double simTime, double dt) {
   int status = PV_SUCCESS;
   if (needUpdate(simTime, dt)) {
      if (needReset(simTime, dt)) {
         status           = resetStateOnTrigger();
         mLastTriggerTime = simTime;
      }

      update_timer->start();
#ifdef PV_USE_CUDA
      if (mUpdateGpu) {
         gpu_update_timer->start();
         float *gSynHead = GSyn == NULL ? NULL : GSyn[0];
         assert(mUpdateGpu);
         status = updateStateGpu(simTime, dt);
         gpu_update_timer->stop();
      }
      else {
#endif
         status = updateState(simTime, dt);
#ifdef PV_USE_CUDA
      }
      // Activity updated, set flag to true
      updatedDeviceActivity  = true;
      updatedDeviceDatastore = true;
#endif
      update_timer->stop();
      mNeedToPublish  = true;
      mLastUpdateTime = simTime;
   }
   return status;
}

int HyPerLayer::resetStateOnTrigger() {
   assert(triggerResetLayer != NULL);
   float *V = getV();
   if (V == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: triggerBehavior is \"resetStateOnTrigger\" but layer does not have a membrane "
               "potential.\n",
               getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   float const *resetV = triggerResetLayer->getV();
   if (resetV != NULL) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
      for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
         V[k] = resetV[k];
      }
   }
   else {
      float const *resetA   = triggerResetLayer->getActivity();
      PVLayerLoc const *loc = triggerResetLayer->getLayerLoc();
      PVHalo const *halo    = &loc->halo;
      for (int b = 0; b < parent->getNBatch(); b++) {
         float const *resetABatch = resetA + (b * triggerResetLayer->getNumExtended());
         float *VBatch            = V + (b * triggerResetLayer->getNumNeurons());
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getNumNeurons(); k++) {
            int kex = kIndexExtended(
                  k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
            VBatch[k] = resetABatch[kex];
         }
      }
   }

   int status = setActivity();

// Update V on GPU after CPU V gets set
#ifdef PV_USE_CUDA
   if (mUpdateGpu) {
      getDeviceV()->copyToDevice(V);
      // Right now, we're setting the activity on the CPU and memsetting the GPU memory
      // TODO calculate this on the GPU
      getDeviceActivity()->copyToDevice(clayer->activity->data);
      // We need to updateDeviceActivity and Datastore if we're resetting V
      updatedDeviceActivity  = true;
      updatedDeviceDatastore = true;
   }
#endif

   return status;
}

int HyPerLayer::resetGSynBuffers(double timef, double dt) {
   int status = PV_SUCCESS;
   if (GSyn == NULL)
      return PV_SUCCESS;
   resetGSynBuffers_HyPerLayer(
         parent->getNBatch(), this->getNumNeurons(), getNumChannels(), GSyn[0]);
   return status;
}

#ifdef PV_USE_CUDA
int HyPerLayer::runUpdateKernel() {

#ifdef PV_USE_CUDA
   assert(mUpdateGpu);
   if (updatedDeviceGSyn) {
      copyAllGSynToDevice();
      updatedDeviceGSyn = false;
   }

   // V and Activity are write only buffers, so we don't need to do anything with them
   assert(krUpdate);

   // Sync all buffers before running
   syncGpu();

   // Run kernel
   krUpdate->run();
#endif

   return PV_SUCCESS;
}

int HyPerLayer::updateStateGpu(double timef, double dt) {
   Fatal() << "Update state for layer " << name << " is not implemented\n";
   return -1;
}
#endif

int HyPerLayer::updateState(double timef, double dt) {
   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   const PVLayerLoc *loc = getLayerLoc();
   float *A              = getCLayer()->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = GSyn == NULL ? NULL : GSyn[0];

   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int nbatch      = loc->nbatch;
   int num_neurons = nx * ny * nf;
   if (num_channels == 1) {
      applyGSyn_HyPerLayer1Channel(nbatch, num_neurons, V, gSynHead);
   }
   else {
      applyGSyn_HyPerLayer(nbatch, num_neurons, V, gSynHead);
   }
   setActivity_HyPerLayer(
         nbatch,
         num_neurons,
         A,
         V,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up);

   return PV_SUCCESS;
}

int HyPerLayer::setActivity() {
   const PVLayerLoc *loc = getLayerLoc();
   return setActivity_HyPerLayer(
         loc->nbatch,
         getNumNeurons(),
         clayer->activity->data,
         getV(),
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up);
}

// Updates active indices for all levels (delays) here
int HyPerLayer::updateAllActiveIndices() { return publisher->updateAllActiveIndices(); }
int HyPerLayer::updateActiveIndices() { return publisher->updateActiveIndices(0); }

bool HyPerLayer::isExchangeFinished(int delay) { return publisher->isExchangeFinished(delay); }

bool HyPerLayer::isAllInputReady() {
   bool isReady = true;
   for (auto &c : recvConns) {
      for (int a = 0; a < c->numberOfAxonalArborLists(); a++) {
         isReady &= c->getPre()->isExchangeFinished(c->getDelay(a));
      }
   }
   return isReady;
}

int HyPerLayer::recvAllSynapticInput() {
   int status = PV_SUCCESS;
   // Only recvAllSynapticInput if we need an update
   if (needUpdate(parent->simulationTime(), parent->getDeltaTime())) {
      bool switchGpu = false;
      // Start CPU timer here
      recvsyn_timer->start();

      for (auto &conn : recvConns) {
         pvAssert(conn != NULL);
#ifdef PV_USE_CUDA
         // Check if it's done with cpu connections
         if (!switchGpu && conn->getReceiveGpu()) {
            // Copy GSyn over to GPU
            copyAllGSynToDevice();
            // Start gpu timer
            gpu_recvsyn_timer->start();
            switchGpu = true;
         }
#endif
         conn->deliver();
      }
#ifdef PV_USE_CUDA
      if (switchGpu) {
         // Stop timer
         gpu_recvsyn_timer->stop();
      }
#endif
      recvsyn_timer->stop();
   }
   return status;
}

#ifdef PV_USE_CUDA
double HyPerLayer::addGpuTimers() {
   double simTime    = 0;
   bool updateNeeded = needUpdate(parent->simulationTime(), parent->getDeltaTime());
   if (mRecvGpu && updateNeeded) {
      simTime += gpu_recvsyn_timer->accumulateTime();
   }
   if (mUpdateGpu && updateNeeded) {
      simTime += gpu_update_timer->accumulateTime();
   }
   return simTime;
}

void HyPerLayer::syncGpu() {
   if (mRecvGpu || mUpdateGpu) {
      parent->getDevice()->syncDevice();
   }
}

void HyPerLayer::copyAllGSynToDevice() {
   if (mRecvGpu || mUpdateGpu) {
      // Copy it to device
      // Allocated as a big chunk, this should work
      float *h_postGSyn              = GSyn[0];
      PVCuda::CudaBuffer *d_postGSyn = this->getDeviceGSyn();
      assert(d_postGSyn);
      d_postGSyn->copyToDevice(h_postGSyn);
   }
}

void HyPerLayer::copyAllGSynFromDevice() {
   // Only copy if recving
   if (mRecvGpu) {
      // Allocated as a big chunk, this should work
      float *h_postGSyn              = GSyn[0];
      PVCuda::CudaBuffer *d_postGSyn = this->getDeviceGSyn();
      assert(d_postGSyn);
      d_postGSyn->copyFromDevice(h_postGSyn);
   }
}

void HyPerLayer::copyAllVFromDevice() {
   // Only copy if updating
   if (mUpdateGpu) {
      // Allocated as a big chunk, this should work
      float *h_V              = getV();
      PVCuda::CudaBuffer *d_V = this->getDeviceV();
      assert(d_V);
      d_V->copyFromDevice(h_V);
   }
}

void HyPerLayer::copyAllActivityFromDevice() {
   // Only copy if updating
   if (mUpdateGpu) {
      // Allocated as a big chunk, this should work
      float *h_activity              = getCLayer()->activity->data;
      PVCuda::CudaBuffer *d_activity = this->getDeviceActivity();
      assert(d_activity);
      d_activity->copyFromDevice(h_activity);
   }
}

#endif

int HyPerLayer::publish(Communicator *comm, double simTime) {
   publish_timer->start();

   int status = PV_SUCCESS;
   if (mNeedToPublish) {
      if (useMirrorBCs()) {
         mirrorInteriorToBorder(clayer->activity, clayer->activity);
      }
      status         = publisher->publish(mLastUpdateTime);
      mNeedToPublish = false;
   }
   else {
      publisher->copyForward(mLastUpdateTime);
   }
   publish_timer->stop();
   return status;
}

int HyPerLayer::waitOnPublish(Communicator *comm) {
   publish_timer->start();

   // wait for MPI border transfers to complete
   //
   int status = publisher->wait();

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
int HyPerLayer::insertProbe(LayerProbe *p) {
   if (p->getTargetLayer() != this) {
      WarnLog().printf(
            "HyPerLayer \"%s\": insertProbe called with probe %p, whose targetLayer is not this "
            "layer.  Probe was not inserted.\n",
            name,
            p);
      return numProbes;
   }
   for (int i = 0; i < numProbes; i++) {
      if (p == probes[i]) {
         WarnLog().printf(
               "HyPerLayer \"%s\": insertProbe called with probe %p, which has already been "
               "inserted as probe %d.\n",
               name,
               p,
               i);
         return numProbes;
      }
   }

   // malloc'ing a new buffer, copying data over, and freeing the old buffer could be replaced by
   // malloc
   LayerProbe **tmp;
   tmp = (LayerProbe **)malloc((numProbes + 1) * sizeof(LayerProbe *));
   assert(tmp != NULL);

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   free(probes);

   probes            = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerLayer::outputProbeParams() {
   int status = PV_SUCCESS;
   for (int p = 0; p < numProbes; p++) {
      int status1 = probes[p]->writeParams();
      if (status1 != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

int HyPerLayer::outputState(double timef) {
   int status = PV_SUCCESS;

   io_timer->start();

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputStateWrapper(timef, parent->getDeltaTime());
   }

   if (timef >= (writeTime - (parent->getDeltaTime() / 2)) && writeStep >= 0) {
      writeTime += writeStep;
      if (sparseLayer) {
         status = writeActivitySparse(timef);
      }
      else {
         status = writeActivity(timef);
      }
   }
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "%s: outputState failed on rank %d process.\n", getDescription_c(), parent->columnId());
   }

   io_timer->stop();
   return status;
}

int HyPerLayer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   int status = PV_SUCCESS;
   if (initializeFromCheckpointFlag) {
      status = readActivityFromCheckpoint(checkpointer);
      status = readVFromCheckpoint(checkpointer);
      status = readDelaysFromCheckpoint(checkpointer);
      updateAllActiveIndices();
   }
   return status;
}

int HyPerLayer::readActivityFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), std::string("A"), false);
   return PV_SUCCESS;
}

int HyPerLayer::readVFromCheckpoint(Checkpointer *checkpointer) {
   if (getV() != nullptr) {
      checkpointer->readNamedCheckpointEntry(std::string(name), std::string("V"), false);
   }
   return PV_SUCCESS;
}

int HyPerLayer::readDelaysFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), std::string("Delays"), false);
   return PV_SUCCESS;
}

// readBufferFile and readDataStoreFromFile were removed Jan 23, 2017.
// They were only used by checkpointing, which is now handled by the
// CheckpointEntry class hierarchy.

int HyPerLayer::processCheckpointRead() { return updateAllActiveIndices(); }

int HyPerLayer::writeActivitySparse(double timed) {
   PVLayerCube cube      = publisher->createCube(0 /*delay*/);
   PVLayerLoc const *loc = getLayerLoc();
   pvAssert(cube.numItems == loc->nbatch * getNumExtended());

   int const mpiBatchDimension = getMPIBlock()->getBatchDimension();
   int const numFrames         = mpiBatchDimension * loc->nbatch;
   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % loc->nbatch;
      int const mpiBatchIndex   = frame / loc->nbatch; // Integer division
      pvAssert(mpiBatchIndex * loc->nbatch + localBatchIndex == frame);

      SparseList<float> list;
      auto *activeIndicesBatch   = (SparseList<float>::Entry const *)cube.activeIndices;
      auto *activeIndicesElement = &activeIndicesBatch[localBatchIndex * getNumExtended()];
      PVLayerLoc const *loc      = getLayerLoc();
      int nxExt                  = loc->nx + loc->halo.lt + loc->halo.rt;
      int nyExt                  = loc->ny + loc->halo.dn + loc->halo.up;
      int nf                     = loc->nf;
      for (long int k = 0; k < cube.numActive[localBatchIndex]; k++) {
         SparseList<float>::Entry entry = activeIndicesElement[k];
         int index                      = (int)entry.index;

         // Location is local extended; need global restricted.
         // Get local restricted coordinates.
         int x = kxPos(index, nxExt, nyExt, nf) - loc->halo.lt;
         if (x < 0 or x >= loc->nx) {
            continue;
         }
         int y = kyPos(index, nxExt, nyExt, nf) - loc->halo.up;
         if (y < 0 or y >= loc->ny) {
            continue;
         }
         // Convert to global restricted coordinates.
         x += loc->kx0;
         y += loc->ky0;
         int f = featureIndex(index, nxExt, nyExt, nf);

         // Get global restricted index.
         entry.index = (uint32_t)kIndex(x, y, f, loc->nxGlobal, loc->nyGlobal, nf);
         list.addEntry(entry);
      }
      auto gatheredList =
            BufferUtils::gatherSparse(getMPIBlock(), list, mpiBatchIndex, 0 /*root process*/);
      if (getMPIBlock()->getRank() == 0) {
         long fpos = mOutputStateStream->getOutPos();
         if (fpos == 0L) {
            BufferUtils::ActivityHeader header = BufferUtils::buildSparseActivityHeader<float>(
                  loc->nx * getMPIBlock()->getNumColumns(),
                  loc->ny * getMPIBlock()->getNumRows(),
                  loc->nf,
                  0 /* numBands */); // numBands will be set by call to incrementNBands.
            header.timestamp = timed;
            BufferUtils::writeActivityHeader(*mOutputStateStream, header);
         }
         BufferUtils::writeSparseFrame(*mOutputStateStream, &gatheredList, timed);
      }
   }
   writeActivitySparseCalls += numFrames;
   updateNBands(writeActivitySparseCalls);
   return PV_SUCCESS;
}

// write non-spiking activity
int HyPerLayer::writeActivity(double timed) {
   PVLayerCube cube      = publisher->createCube(0);
   PVLayerLoc const *loc = getLayerLoc();
   pvAssert(cube.numItems == loc->nbatch * getNumExtended());

   PVHalo const &halo   = loc->halo;
   int const nxExtLocal = loc->nx + halo.lt + halo.rt;
   int const nyExtLocal = loc->ny + halo.dn + halo.up;
   int const nf         = loc->nf;

   int const mpiBatchDimension = getMPIBlock()->getBatchDimension();
   int const numFrames         = mpiBatchDimension * loc->nbatch;
   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % loc->nbatch;
      int const mpiBatchIndex   = frame / loc->nbatch; // Integer division
      pvAssert(mpiBatchIndex * loc->nbatch + localBatchIndex == frame);

      float *data = &cube.data[localBatchIndex * getNumExtended()];
      Buffer<float> localBuffer(data, nxExtLocal, nyExtLocal, nf);
      localBuffer.crop(loc->nx, loc->ny, Buffer<float>::CENTER);
      Buffer<float> blockBuffer = BufferUtils::gather<float>(
            getMPIBlock(), localBuffer, loc->nx, loc->ny, mpiBatchIndex, 0 /*root process*/);
      // At this point, the rank-zero process has the entire block for the batch element,
      // regardless of what the mpiBatchIndex is.
      if (getMPIBlock()->getRank() == 0) {
         long fpos = mOutputStateStream->getOutPos();
         if (fpos == 0L) {
            BufferUtils::ActivityHeader header = BufferUtils::buildActivityHeader<float>(
                  loc->nx * getMPIBlock()->getNumColumns(),
                  loc->ny * getMPIBlock()->getNumRows(),
                  loc->nf,
                  0 /* numBands */); // numBands will be set by call to incrementNBands.
            header.timestamp = timed;
            BufferUtils::writeActivityHeader(*mOutputStateStream, header);
         }
         BufferUtils::writeFrame<float>(*mOutputStateStream, &blockBuffer, timed);
      }
   }
   writeActivityCalls += numFrames;
   updateNBands(writeActivityCalls);
   return PV_SUCCESS;
}

void HyPerLayer::updateNBands(int const numCalls) {
   // Only the root process needs to maintain INDEX_NBANDS, so only the root process modifies
   // numCalls
   // This way, writeActivityCalls does not need to be coordinated across MPI
   if (mOutputStateStream != nullptr) {
      long int fpos = mOutputStateStream->getOutPos();
      mOutputStateStream->setOutPos(sizeof(int) * INDEX_NBANDS, true /*fromBeginning*/);
      mOutputStateStream->write(&numCalls, (long)sizeof(numCalls));
      mOutputStateStream->setOutPos(fpos, true /*fromBeginning*/);
   }
}

bool HyPerLayer::localDimensionsEqual(PVLayerLoc const *loc1, PVLayerLoc const *loc2) {
   return loc1->nbatch == loc2->nbatch && loc1->nx == loc2->nx && loc1->ny == loc2->ny
          && loc1->nf == loc2->nf && loc1->halo.lt == loc2->halo.lt
          && loc1->halo.rt == loc2->halo.rt && loc1->halo.dn == loc2->halo.dn
          && loc1->halo.up == loc2->halo.up;
}

int HyPerLayer::mirrorToNorthWest(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int nbatch     = dest->loc.nbatch;
   int nf         = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder  = dest->loc.halo.up;
   size_t sb      = strideBExtended(&dest->loc);
   size_t sf      = strideFExtended(&dest->loc);
   size_t sx      = strideXExtended(&dest->loc);
   size_t sy      = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;

      float *src0 = srcData + topBorder * sy + leftBorder * sx;
      float *dst0 = srcData + (topBorder - 1) * sy + (leftBorder - 1) * sx;

      for (int ky = 0; ky < topBorder; ky++) {
         float *to   = dst0 - ky * sy;
         float *from = src0 + ky * sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToNorth(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int nx         = clayer->loc.nx;
   int nf         = clayer->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder  = dest->loc.halo.up;
   int nbatch     = dest->loc.nbatch;
   size_t sb      = strideBExtended(&dest->loc);
   size_t sf      = strideFExtended(&dest->loc);
   size_t sx      = strideXExtended(&dest->loc);
   size_t sy      = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;
      float *src0     = srcData + topBorder * sy + leftBorder * sx;
      float *dst0     = destData + (topBorder - 1) * sy + leftBorder * sx;

      for (int ky = 0; ky < topBorder; ky++) {
         float *to   = dst0 - ky * sy;
         float *from = src0 + ky * sy;
         for (int kx = 0; kx < nx; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToNorthEast(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int nx          = dest->loc.nx;
   int nf          = dest->loc.nf;
   int leftBorder  = dest->loc.halo.lt;
   int rightBorder = dest->loc.halo.rt;
   int topBorder   = dest->loc.halo.up;
   int nbatch      = dest->loc.nbatch;
   size_t sb       = strideBExtended(&dest->loc);
   size_t sf       = strideFExtended(&dest->loc);
   size_t sx       = strideXExtended(&dest->loc);
   size_t sy       = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;
      float *src0     = srcData + topBorder * sy + (nx + leftBorder - 1) * sx;
      float *dst0     = destData + (topBorder - 1) * sy + (nx + leftBorder) * sx;

      for (int ky = 0; ky < topBorder; ky++) {
         float *to   = dst0 - ky * sy;
         float *from = src0 + ky * sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToWest(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int ny         = dest->loc.ny;
   int nf         = dest->loc.nf;
   int leftBorder = dest->loc.halo.lt;
   int topBorder  = dest->loc.halo.up;
   int nbatch     = dest->loc.nbatch;
   size_t sb      = strideBExtended(&dest->loc);
   size_t sf      = strideFExtended(&dest->loc);
   size_t sx      = strideXExtended(&dest->loc);
   size_t sy      = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;
      float *src0     = srcData + topBorder * sy + leftBorder * sx;
      float *dst0     = destData + topBorder * sy + (leftBorder - 1) * sx;

      for (int ky = 0; ky < ny; ky++) {
         float *to   = dst0 + ky * sy;
         float *from = src0 + ky * sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToEast(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int nx          = clayer->loc.nx;
   int ny          = clayer->loc.ny;
   int nf          = clayer->loc.nf;
   int leftBorder  = dest->loc.halo.lt;
   int rightBorder = dest->loc.halo.rt;
   int topBorder   = dest->loc.halo.up;
   int nbatch      = dest->loc.nbatch;
   size_t sb       = strideBExtended(&dest->loc);
   size_t sf       = strideFExtended(&dest->loc);
   size_t sx       = strideXExtended(&dest->loc);
   size_t sy       = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;
      float *src0     = srcData + topBorder * sy + (nx + leftBorder - 1) * sx;
      float *dst0     = destData + topBorder * sy + (nx + leftBorder) * sx;

      for (int ky = 0; ky < ny; ky++) {
         float *to   = dst0 + ky * sy;
         float *from = src0 + ky * sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToSouthWest(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int ny           = dest->loc.ny;
   int nf           = dest->loc.nf;
   int leftBorder   = dest->loc.halo.lt;
   int topBorder    = dest->loc.halo.up;
   int bottomBorder = dest->loc.halo.dn;
   int nbatch       = dest->loc.nbatch;
   size_t sb        = strideBExtended(&dest->loc);
   size_t sf        = strideFExtended(&dest->loc);
   size_t sx        = strideXExtended(&dest->loc);
   size_t sy        = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;
      float *src0     = srcData + (ny + topBorder - 1) * sy + leftBorder * sx;
      float *dst0     = destData + (ny + topBorder) * sy + (leftBorder - 1) * sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         float *to   = dst0 + ky * sy;
         float *from = src0 - ky * sy;
         for (int kx = 0; kx < leftBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to -= nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToSouth(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int nx           = dest->loc.nx;
   int ny           = dest->loc.ny;
   int nf           = dest->loc.nf;
   int leftBorder   = dest->loc.halo.lt;
   int rightBorder  = dest->loc.halo.rt;
   int topBorder    = dest->loc.halo.up;
   int bottomBorder = dest->loc.halo.dn;
   int nbatch       = dest->loc.nbatch;
   size_t sb        = strideBExtended(&dest->loc);
   size_t sf        = strideFExtended(&dest->loc);
   size_t sx        = strideXExtended(&dest->loc);
   size_t sy        = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;
      float *src0     = srcData + (ny + topBorder - 1) * sy + leftBorder * sx;
      float *dst0     = destData + (ny + topBorder) * sy + leftBorder * sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         float *to   = dst0 + ky * sy;
         float *from = src0 - ky * sy;
         for (int kx = 0; kx < nx; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from += nf;
         }
      }
   }
   return 0;
}

int HyPerLayer::mirrorToSouthEast(PVLayerCube *dest, PVLayerCube *src) {
   if (!localDimensionsEqual(&dest->loc, &src->loc)) {
      return -1;
   }
   int nx           = dest->loc.nx;
   int ny           = dest->loc.ny;
   int nf           = dest->loc.nf;
   int leftBorder   = dest->loc.halo.lt;
   int rightBorder  = dest->loc.halo.rt;
   int topBorder    = dest->loc.halo.up;
   int bottomBorder = dest->loc.halo.dn;
   int nbatch       = dest->loc.nbatch;
   size_t sb        = strideBExtended(&dest->loc);
   size_t sf        = strideFExtended(&dest->loc);
   size_t sx        = strideXExtended(&dest->loc);
   size_t sy        = strideYExtended(&dest->loc);

   for (int b = 0; b < nbatch; b++) {
      float *srcData  = src->data + b * sb;
      float *destData = dest->data + b * sb;
      float *src0     = srcData + (ny + topBorder - 1) * sy + (nx + leftBorder - 1) * sx;
      float *dst0     = destData + (ny + topBorder) * sy + (nx + leftBorder) * sx;

      for (int ky = 0; ky < bottomBorder; ky++) {
         float *to   = dst0 + ky * sy;
         float *from = src0 - ky * sy;
         for (int kx = 0; kx < rightBorder; kx++) {
            for (int kf = 0; kf < nf; kf++) {
               to[kf * sf] = from[kf * sf];
            }
            to += nf;
            from -= nf;
         }
      }
   }
   return 0;
}

} // end of PV namespace
