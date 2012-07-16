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
 */

#include "HyPerLayer.hpp"
#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../columns/HyPerCol.hpp"
#include "../connections/HyPerConn.hpp"
#include "InitV.hpp"
#include "../io/fileio.hpp"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

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
   this->numProbes = 0;
   this->ioAppend = 0;
   this->numChannels = 0;
   this->clayer = NULL;
   this->GSyn = NULL;
   this->labels = NULL;
   this->marginIndices = NULL;
   this->numMargin = 0;
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
   this->update_timer = NULL;
   this->recvsyn_timer = NULL;
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

   PVParams * params = parent->parameters();

   if (params->present(name, "nx") || params->present(name, "ny")) {
      fprintf(stderr, "HyPerLayer::initialize_base: ERROR, use (nxScale,nyScale) not (nx,ny)\n");
      exit(-1);
   }

   const float nxScale = params->value(name, "nxScale", 1.0f);
   const float nyScale = params->value(name, "nyScale", 1.0f);

   const int numFeatures = (int) params->value(name, "nf", 1);
   const int margin      = (int) params->value(name, "marginWidth", 0);

   double xScaled = -log2( (double) nxScale);
   double yScaled = -log2( (double) nyScale);

   int xScale = (int) nearbyint(xScaled);
   int yScale = (int) nearbyint(yScaled);

   writeTime = parent->simulationTime();
   writeStep = params->value(name, "writeStep", parent->getDeltaTime());

#undef WRITE_NONSPIKING_ACTIVITY
#ifdef WRITE_NONSPIKING_ACTIVITY
   float defaultWriteNonspikingActivity = 1.0;
#else
   float defaultWriteNonspikingActivity = 0.0;
#endif

   spikingFlag = (bool) params->value(name, "spikingFlag", 0);
   if( !spikingFlag )
      writeNonspikingActivity = (bool) params->value(name,
         "writeNonspikingActivity", defaultWriteNonspikingActivity);

   writeActivityCalls = 0;
   writeActivitySparseCalls = 0;

   mirrorBCflag = (bool) params->value(name, "mirrorBCflag", 0);

   PVLayerLoc layerLoc;
   setLayerLoc(&layerLoc, nxScale, nyScale, margin, numFeatures);
   clayer = pvlayer_new(layerLoc, xScale, yScale, numChannels);
   clayer->layerType = TypeGeneric;
   // layerId stored as clayer->layerId
   int layerID = parent->addLayer(this); // Could this line and the setParent line be combined in a HyPerLayer method?
   assert(layerID == clayer->layerId);

   // allocate storage for the input conductance arrays
   //
   int status = allocateBuffers();
   assert(status == PV_SUCCESS);

   // Initializing now takes place at the beginning of HyPerCol::run(int), after
   // the publishers have been initialized, to allow loading data into the datastore
#ifdef OBSOLETE // Marked obsolete July 11, 2012
   bool restart_flag = params->value(name, "restart", 0.0f) != 0.0f;
   if( restart_flag ) {
      float timef;
      readState(&timef);
   }
   else {
      initializeState();
   }
#endif // OBSOLETE

   // labels are not extended
   labels = (int *) calloc(getNumNeurons(), sizeof(int));
   assert(labels != NULL);

#ifdef PV_USE_OPENCL
   initUseGPUFlag();
#endif
   return PV_SUCCESS;
}

#ifdef PV_USE_OPENCL
//This method checks for a parameter telling Petavision to GPU accellerate
//this layer
void HyPerLayer::initUseGPUFlag() {
   PVParams * params = parent->parameters();
   gpuAccelerateFlag = params->value(name, "GPUAccelerate", gpuAccelerateFlag);
   copyDataStoreFlag=false;
   //buffersInitialized=false;
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
      fflush(stdout);
   }
   delete recvsyn_timer; recvsyn_timer = NULL;
   delete update_timer; update_timer = NULL;

   if (clayer != NULL) {
      // pvlayer_finalize will free clayer
      pvlayer_finalize(clayer);
      clayer = NULL;
   }
   
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

   if (numChannels > 0) {
      free(GSyn[0]);  // conductances allocated contiguously so frees all buffer storage
      free(GSyn);     // this frees the array pointers to separate conductance channels
      GSyn = NULL;
      numChannels = 0;
   }
}

#ifdef PV_USE_OPENCL
#endif

/**
 * Initialize a few things that require a layer id
 */
int HyPerLayer::initializeLayerId(int layerId)
{
   char filename[PV_PATH_MAX];
   bool append = false;

   setLayerId(layerId);
   switch( parent->includeLayerName() ) {
   case 0:
      snprintf(filename, PV_PATH_MAX, "%s/a%d.pvp", parent->getOutputPath(), clayer->layerId);
      break;
   case 1:
      snprintf(filename, PV_PATH_MAX, "%s/a%d_%s.pvp", parent->getOutputPath(), clayer->layerId, name);
      break;
   case 2:
      snprintf(filename, PV_PATH_MAX, "%s/%s.pvp", parent->getOutputPath(), name);
      break;
   default:
      assert(0);
      break;
   }
   clayer->activeFP = pvp_open_write_file(filename, parent->icCommunicator(), append);

   return 0;
}

int HyPerLayer::setLayerLoc(PVLayerLoc * layerLoc, float nxScale, float nyScale, int margin, int nf)
{
   InterColComm * icComm = parent->icCommunicator();
   layerLoc->nxGlobal = (int) (nxScale * parent->getNxGlobal());
   layerLoc->nyGlobal = (int) (nyScale * parent->getNyGlobal());

   // partition input space based on the number of processor
   // columns and rows
   //

   layerLoc->nx = layerLoc->nxGlobal / icComm->numCommColumns();
   layerLoc->ny = layerLoc->nyGlobal / icComm->numCommRows();

   assert(layerLoc->nxGlobal == layerLoc->nx * icComm->numCommColumns());
   assert(layerLoc->nyGlobal == layerLoc->ny * icComm->numCommRows());

   layerLoc->kx0 = layerLoc->nx * icComm->commColumn();
   layerLoc->ky0 = layerLoc->ny * icComm->commRow();

   layerLoc->nf = nf;
   layerLoc->nb = margin;

   layerLoc->halo.lt = margin;
   layerLoc->halo.rt = margin;
   layerLoc->halo.dn = margin;
   layerLoc->halo.up = margin;

   return 0;
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
   bool restart_flag = params->value(name, "restart", 0.0f) != 0.0f;
   if( restart_flag ) {
      float timef;
      status = readState(&timef);
   }
   else {
      InitV * initVObject = new InitV(parent, name);
      if( initVObject == NULL ) {
         fprintf(stderr, "HyPerLayer::initializeState error: layer %s unable to create InitV object\n", name);
         abort();
      }
      status = initVObject->calcV(this);
      delete initVObject;
      setActivity();
   }
   return status;
}

#ifdef PV_USE_OPENCL
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

int HyPerLayer::columnWillAddLayer(InterColComm * comm, int layerId)
{
   clayer->columnId = parent->columnId();
   initializeLayerId(layerId);

   // addPublisher call has been moved to start of HyPerCol::run(int), so that connections can adjust numDelayLevels as necessary.
   // comm->addPublisher(this, clayer->activity->numItems, clayer->numDelayLevels);

   return 0;
}

int HyPerLayer::initFinish()
{
   return 0;
}

/*
 * Call this routine to increase the number of levels in the data store ring buffer.
 * Calls to this routine after the data store has been initialized will have no effect.
 * The routine returns the new value of clayer->numDelayLevels
 */
int HyPerLayer::increaseDelayLevels(int neededDelay) {
   if( clayer->numDelayLevels < neededDelay+1 ) clayer->numDelayLevels = neededDelay+1;
   if( clayer->numDelayLevels > MAX_F_DELAY ) clayer->numDelayLevels = MAX_F_DELAY;
   return clayer->numDelayLevels;
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
//   if(!buffersInitialized) {
//      //this may seem like a strange place to do this, but when the
//      //layer is being created, the publishers don't exist yet!
//      if(initializeDataStoreThreadBuffers()) {
//         buffersInitialized=true;
//      }
//      //else
//        // return NULL;
//   }

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

int HyPerLayer::updateState(float timef, float dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], getSpikingFlag(), getCLayer()->activeIndices, &getCLayer()->numActive);
   if(status == PV_SUCCESS) status = updateActiveIndices();
   return status;
}

int HyPerLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active)
{
   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   //pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
   //pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);
   updateV_HyPerLayer(num_neurons, V, gSynHead);
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->nb);
   // setActivity();
   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead); // resetGSynBuffers();

   return PV_SUCCESS;
}

int HyPerLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_HyPerLayer(getNumNeurons(), clayer->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb);
}

int HyPerLayer::updateBorder(float time, float dt)
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
   if( spikingFlag ) return calcActiveIndices(); else return PV_SUCCESS;
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
#endif


   for (int kPre = 0; kPre < numExtended; kPre++) {
      float a = activity->data[kPre];
      // Activity < 0 is used by generative models --pete
      if (a == 0.0f) continue;  // TODO - assume activity is sparse so make this common branch

      PVPatch * weights = conn->getWeights(kPre, arborID);

      // WARNING - assumes weight and GSyn patches from task same size
      //         - assumes patch stride sf is 1

      int nk  = conn->fPatchSize() * weights->nx;
      int ny  = weights->ny;
      int sy  = conn->getPostNonextStrides()->sy;       // stride in layer
      int syw = conn->yPatchStride(); //weights->sy;    // stride in patch
      pvdata_t * gSynPatchStart = conn->getGSynPatchStart(kPre, arborID);
      // TODO - unroll
      //int patchSize = conn->xPatchSize()*conn->yPatchSize()*conn->fPatchSize();
      //pvdata_t * data = conn->get_wDataHead(arborID, conn->correctPIndex(kPre)) + weights->offset;
      //pvdata_t * data = conn->get_wDataStart(arborID) + conn->correctPIndex(kPre)*patchSize + weights->offset;
      // int patchSize = conn->xPatchSize()*conn->yPatchSize()*conn->fPatchSize();
      pvdata_t * data = conn->get_wData(arborID,kPre);
      for (int y = 0; y < ny; y++) {
         (conn->accumulateFunctionPointer)(nk, gSynPatchStart + y*sy, a, data + y*syw);
//       if (err != 0) printf("  ERROR kPre = %d\n", kPre);
      }
   }

   recvsyn_timer->stop();

   return 0;
}

int HyPerLayer::reconstruct(HyPerConn * conn, PVLayerCube * cube)
{
   // TODO - implement
   printf("[%d]: HyPerLayer::reconstruct: to layer %d from %d\n",
          clayer->columnId, clayer->layerId, conn->preSynapticLayer()->clayer->layerId);
   return 0;
}

int HyPerLayer::triggerReceive(InterColComm* comm)
{
   // deliver calls recvSynapticInput for all presynaptic connections
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

int HyPerLayer::publish(InterColComm* comm, float time)
{
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
   return status;
}

int HyPerLayer::waitOnPublish(InterColComm* comm)
{
   // wait for MPI border transfers to complete
   //
   int status = comm->wait(getLayerId());
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
   delete probes;

   probes = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerLayer::outputState(float timef, bool last)
{
   int status = PV_SUCCESS;

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(timef);
   }


   if (timef >= writeTime && writeStep >= 0) {
      writeTime += writeStep;
      if (spikingFlag != 0) {
         status = writeActivitySparse(timef);
      }
      else {
         if (writeNonspikingActivity) {
            status = writeActivity(timef);
         }
      }
   }

   return status;
}

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

int HyPerLayer::checkpointRead(const char * cpDir, float * timef) {
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
   double timed;
   assert(filename != NULL);
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_A.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   readBufferFile(filename, icComm, &timed, clayer->activity->data, 1, /*extended*/true, /*contiguous*/false);
   *timef = (float) timed;
   // TODO contiguous should be true in the writeBufferFile calls (needs to be added to writeBuffer method)
   if( getV() != NULL ) {
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s_V.pvp", basepath);
      assert(chars_needed < PV_PATH_MAX);
      readBufferFile(filename, icComm, &timed, getV(), 1, /*extended*/false, /*contiguous*/false);
      if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
         fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
      }
   }
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Delays.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   readDataStoreFromFile(filename, icComm, &timed);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_nextWrite.bin", basepath);
   assert(chars_needed < PV_PATH_MAX);
   if( parent->icCommunicator()->commRank() == 0 ) {
      FILE * fpWriteTime = fopen(filename, "r");
      pvdata_t write_time = writeTime;
      if (fpWriteTime==NULL  && parent->icCommunicator()->commRank() == 0 ) {
         fprintf(stderr, "HyPerLayer::checkpointRead warning: unable to open path %s for reading.  writeTime will be %f\n", filename, write_time);
      }
      else {
         int num_read = fread(&writeTime, sizeof(writeTime), 1, fpWriteTime);
         if (num_read != 1 && parent->icCommunicator()->commRank() == 0 ) {
            fprintf(stderr, "HyPerLayer::checkpointRead warning: unable to read from %s.  writeTime will be %f\n", filename, write_time);
            writeTime = write_time;
         }
      }
      fclose(fpWriteTime);
   }
   MPI_Bcast(&writeTime, 1, MPI_FLOAT, 0, icComm->communicator());

   return PV_SUCCESS;
}

int HyPerLayer::readBufferFile(const char * filename, InterColComm * comm, double * timed, pvdata_t * buffer, int numbands, bool extended, bool contiguous) {
   int status;
   int params[NUM_BIN_PARAMS];
   readHeader(filename, comm, timed, params);
   int buffersize;
   if(extended) {
      buffersize = (getLayerLoc()->nx+2*getLayerLoc()->nb)*(getLayerLoc()->ny+2*getLayerLoc()->nb)*getLayerLoc()->nf;
   }
   else {
      buffersize = getLayerLoc()->nx*getLayerLoc()->ny*getLayerLoc()->nf;
   }
   for( int band=0; band<numbands; band++ ) {
      int status1;
      status1 = readNonspikingActFile(filename, comm, timed, buffer+band*buffersize,
                            band, getLayerLoc(), params[INDEX_DATA_TYPE], extended, contiguous);
      if( status1 != PV_SUCCESS ) {
         status = PV_FAILURE;
      }
   }

   return status;
}

int HyPerLayer::readDataStoreFromFile(const char * filename, InterColComm * comm, double * timeptr) {
   assert(timeptr != NULL);
   int params[NUM_BIN_PARAMS];
   readHeader(filename, comm, timeptr, params);
   assert(params[INDEX_NBANDS] == comm->publisherStore(getCLayer()->layerId)->numberOfLevels());
   assert(params[INDEX_NBANDS] == getCLayer()->numDelayLevels);

   bool contiguous;
   if( params[INDEX_NUM_RECORDS] == comm->numCommColumns()*comm->numCommRows() ) {
      contiguous = false;
   }
   else if( params[INDEX_NUM_RECORDS] == 1 ) {
      contiguous = true;
   }
   else assert(false);
   if( contiguous ) {
      assert(params[INDEX_NX] == getLayerLoc()->nxGlobal);
      assert(params[INDEX_NY] == getLayerLoc()->nyGlobal);
      assert(params[INDEX_RECORD_SIZE] == getNumGlobalNeurons());
      assert(params[INDEX_NX_PROCS] == 1);
      assert(params[INDEX_NY_PROCS] == 1);
   }
   else {
      assert(params[INDEX_NUM_RECORDS] == comm->numCommColumns()*comm->numCommRows());
      assert(params[INDEX_NX] == getLayerLoc()->nx);
      assert(params[INDEX_NY] == getLayerLoc()->ny);
      assert(params[INDEX_RECORD_SIZE] == getNumNeurons());
      assert(params[INDEX_NX_PROCS] == comm->numCommColumns());
      assert(params[INDEX_NY_PROCS] == comm->numCommRows());
   }
   assert(contiguous==false); // TODO contiguous==true case

   DataStore * datastore = comm->publisherStore(getCLayer()->layerId);
   bool extended = true;
   int status = PV_SUCCESS;
   for( int level=0; level<getCLayer()->numDelayLevels; level++ ) {
      pvdata_t * buffer = (pvdata_t *) datastore->buffer(0, level);
      double dummytime;
      int status1;
      status1 = readNonspikingActFile(filename, comm, &dummytime, buffer, level,
                     getLayerLoc(), params[INDEX_DATA_TYPE], extended, contiguous);
      if( status1 != PV_SUCCESS ) status = PV_FAILURE;
      status1 = comm->exchangeBorders(getCLayer()->layerId, getLayerLoc(), level);
   }
   assert( status == PV_SUCCESS);
   return status;
}

int HyPerLayer::readHeader(const char * filename, InterColComm * comm, double * timed, int * params) {
   int filetype, datatype;
   int numParams = NUM_BIN_PARAMS;
   pvp_read_header(filename, comm, timed,
                       &filetype, &datatype, params, &numParams);

   // Sanity checks
   assert(numParams == NUM_BIN_PARAMS);
   assert(params[INDEX_HEADER_SIZE] == NUM_BIN_PARAMS*sizeof(pvdata_t));
   assert(params[INDEX_NUM_PARAMS] == NUM_BIN_PARAMS);
   assert(params[INDEX_FILE_TYPE] == PVP_NONSPIKING_ACT_FILE_TYPE); // TODO allow params[INDEX_FILE_TYPE] == PVP_ACT_FILE_TYPE
   assert(params[INDEX_NF] == getLayerLoc()->nf);
   assert(params[INDEX_DATA_SIZE] == sizeof(pvdata_t));
   assert(params[INDEX_DATA_TYPE] == PV_FLOAT_TYPE);
   assert(params[INDEX_KX0] == 0);
   assert(params[INDEX_KY0] == 0);
   assert(params[INDEX_NB] == getLayerLoc()->nb);
   return PV_SUCCESS;
}

int HyPerLayer::checkpointWrite(const char * cpDir) {
   // Writes checkpoint files for V, A, GSyn(?) and datastore to files in working directory
   // (HyPerCol::checkpointWrite() calls chdir before and after calling this routine)
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
   double timed = (double) parent->simulationTime();
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_A.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, clayer->activity->data, 1, /*extended*/true, /*contiguous*/false);
   // TODO contiguous should be true in the writeBufferFile calls (needs to be added to writeBuffer method)
   if( getV() != NULL ) {
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s_V.pvp", basepath);
      assert(chars_needed < PV_PATH_MAX);
      writeBufferFile(filename, icComm, timed, getV(), 1, /*extended*/false, /*contiguous*/false);
   }
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Delays.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeDataStoreToFile(filename, icComm, timed);

#ifdef OBSOLETE // Marked obsolete Jan 31, 2012.  When checkpointWrite is called, GSyn is blank.  Since GSyn is calculated by triggerReceive, it doesn't need to be saved.
   if( getNumChannels() > 0 ) {
      sprintf(filename, "%s_GSyn.pvp", name);
      writeBufferFile(filename, icComm, timed, GSyn[0], getNumChannels(), /*extended*/false, /*contiguous*/false);
      // assumes GSyn[0], GSyn[1],... are sequential in memory
   }
#endif // OBSOLETE

   if (icComm->commRank()==0) {
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_nextWrite.bin", cpDir, name);
      assert(chars_needed < PV_PATH_MAX);
      FILE * fpWriteTime = fopen(filename, "w");
      if (fpWriteTime==NULL) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error: unable to open path %s for writing.\n", filename);
         abort();
      }
      int num_written = fwrite(&writeTime, sizeof(writeTime), 1, fpWriteTime);
      if (num_written != 1) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error while writing to %s.\n", filename);
         abort();
      }
      fclose(fpWriteTime);
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_nextWrite.txt", cpDir, name);
      assert(chars_needed < PV_PATH_MAX);
      fpWriteTime = fopen(filename, "w");
      if (fpWriteTime==NULL) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error: unable to open path %s for writing.\n", filename);
         abort();
      }
      fprintf(fpWriteTime, "%f\n", writeTime);
      fclose(fpWriteTime);
   }
   return PV_SUCCESS;
}

int HyPerLayer::writeBufferFile(const char * filename, InterColComm * comm, double timed, pvdata_t * buffer, int numbands, bool extended, bool contiguous) {
   FILE * writeFile = pvp_open_write_file(filename, comm, /*append*/false);
   assert( writeFile != NULL || comm->commRank() != 0 );
   int status = writeBuffer(writeFile, comm, timed, buffer, numbands, extended, contiguous);
   pvp_close_file(writeFile, comm);
   writeFile = NULL;
   return status;
}

int HyPerLayer::writeBuffer(FILE * fp, InterColComm * comm, double timed, pvdata_t * buffer, int numbands, bool extended, bool contiguous) {
   assert(contiguous == false); // TODO contiguous == true case

   // write header, but only at the beginning
#ifdef PV_USE_MPI
   int rank = comm->commRank();
#else // PV_USE_MPI
   int rank = 0;
#endif // PV_USE_MPI
   if( rank == 0 ) {
      long fpos = ftell(fp);
      if (fpos == 0L) {
         int status = pvp_write_header(fp, comm, timed, getLayerLoc(), PVP_NONSPIKING_ACT_FILE_TYPE,
                                       PV_FLOAT_TYPE, numbands, extended, contiguous, NUM_BIN_PARAMS, (size_t) getNumNeurons());
         if (status != PV_SUCCESS) return status;
      }
   }

   int buffersize;
   if( extended ) {
      buffersize = (getLayerLoc()->nx+2*getLayerLoc()->nb)*(getLayerLoc()->ny+2*getLayerLoc()->nb)*getLayerLoc()->nf;
   }
   else {
      buffersize = getLayerLoc()->nx*getLayerLoc()->ny*getLayerLoc()->nf;
   }
   int status = PV_SUCCESS;
   for( int band=0; band<numbands; band++ ) {
      if ( rank==0 && fwrite(&timed, sizeof(double), 1, fp) != 1 )              return -1;
      int status1 =  write_pvdata(fp, comm, timed, buffer+band*buffersize, getLayerLoc(), PV_FLOAT_TYPE,
                                  extended, contiguous, PVP_NONSPIKING_ACT_FILE_TYPE);
      status = status1 != PV_SUCCESS ? status1 : status;
   }
   return status;
}

int HyPerLayer::writeDataStoreToFile(const char * filename, InterColComm * comm, double timed) {
   bool extended = true;
   bool contiguous = false;
   int filetype = PVP_NONSPIKING_ACT_FILE_TYPE;
   int datatype = PV_FLOAT_TYPE;
   FILE * writeFile = pvp_open_write_file(filename, comm, /*append*/false);
   assert( writeFile != NULL || comm->commRank() != 0 );
   DataStore * datastore = comm->publisherStore(getCLayer()->layerId);
   int status = PV_SUCCESS;
   int status1;
   status1 = pvp_write_header(writeFile, comm, timed, getLayerLoc(), filetype, datatype,
                              datastore->numberOfLevels(), extended, contiguous,
                              NUM_BIN_PARAMS, (size_t) getNumNeurons());
   if( status1 != PV_SUCCESS ) status = PV_FAILURE;
   for( int level=0; level<clayer->numDelayLevels; level++ ) {
      double leveltime = timed-level*parent->getDeltaTime();
      if ( comm->commRank()==0 ) {
         status1 = fwrite(&leveltime, sizeof(double), 1, writeFile) != 1;
         if( status1 != PV_SUCCESS ) status = PV_FAILURE;
      }
      status1 = write_pvdata(writeFile, comm, leveltime, (pvdata_t *) datastore->buffer(0, level),
                             getLayerLoc(), datatype, extended, contiguous, filetype);
      if( status1 != PV_SUCCESS ) status = PV_FAILURE;
   }
   pvp_close_file(writeFile, comm);
   writeFile = NULL;

   return status;
}

int HyPerLayer::readState(float * timef)
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

#ifdef OBSOLETE // Marked obsolete Jul 13, 2012.  Dumping the state is now done by CheckpointWrite.
int HyPerLayer::writeState(float timef, bool last)
{
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   int status = PV_SUCCESS;

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   const char * last_str = (last) ? "_last" : "";

   if( getV() != NULL ) {
      getOutputFilename(path, "V", last_str);
      status |= write_pvdata(path, comm, timef, getV(), loc, PV_FLOAT_TYPE, extended, contiguous);
   }

   extended = true;
   getOutputFilename(path, "A", last_str);
   status |= write_pvdata(path, comm, timef, getLayerData(), loc, PV_FLOAT_TYPE, extended, contiguous);

   return status;
}
#endif // OBSOLETE

int HyPerLayer::writeActivitySparse(float timef)
{
   int status = PV::writeActivitySparse(clayer->activeFP, parent->icCommunicator(), timef, clayer);
   incrementNBands(&writeActivitySparseCalls);
   return status;
}

// write non-spiking activity
int HyPerLayer::writeActivity(float timef)
{
   // currently numActive only used by writeActivitySparse
   //
   clayer->numActive = 0;

   int status = PV::writeActivity(clayer->activeFP, parent->icCommunicator(), timef, clayer);
   incrementNBands(&writeActivityCalls);
   return status;
}

int HyPerLayer::incrementNBands(int * numCalls) {
   ++*numCalls;
   int status;
   if( parent->icCommunicator()->commRank() == 0 ) {
      long int fpos = ftell(clayer->activeFP);
      fseek(clayer->activeFP, sizeof(int)*INDEX_NBANDS, SEEK_SET);
      int intswritten = fwrite(numCalls, sizeof(int), 1, clayer->activeFP);
      fseek(clayer->activeFP, fpos, SEEK_SET);
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
      int * marginIndices = (int *) calloc(numMargin,
                      sizeof(int));
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

} // end of PV namespace

