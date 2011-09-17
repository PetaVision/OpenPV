/*
 * HyPerLayer.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "HyPerLayer.hpp"
#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../columns/HyPerCol.hpp"
#include "../connections/HyPerConn.hpp"
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
HyPerLayer::HyPerLayer(const char* name, HyPerCol * hc, int numChannels)
{
   initialize_base(name, hc, numChannels);
}

HyPerLayer::~HyPerLayer()
{
   if (parent->columnId() == 0) {
      printf("%32s: total time in %6s %10s: ", name, "layer", "recvsyn");
      recvsyn_timer->elapsed_time();
      printf("%32s: total time in %6s %10s: ", name, "layer", "update ");
      update_timer->elapsed_time();
      fflush(stdout);
   }
   delete recvsyn_timer;
   delete update_timer;

   if (clayer != NULL) {
      // pvlayer_finalize will free clayer
      pvlayer_finalize(clayer);
      clayer = NULL;
   }
   
   free(name);
   freeChannels();
   
#ifdef PV_USE_OPENCL
   delete clV;
   delete clActivity;
   delete clPrevTime;

   if (clGSyn != NULL) {
      for (int m = 0; m < numChannels; m++) {
         delete clGSyn[m];
      }
      free(clGSyn);
      clGSyn = NULL;
   }
#endif

   if (labels != NULL) free(labels);
   if (marginIndices != NULL) free(marginIndices);
}

void HyPerLayer::freeChannels()
{
#ifdef PV_USE_OPENCL
   for (int m = 0; m < numChannels; m++) {
      delete clGSyn[m];
   }
#endif

   if (numChannels > 0) {
      free(GSyn[0]);  // conductances allocated contiguously so frees all buffer storage
      free(GSyn);     // this frees the array pointers to separate conductance channels
      GSyn = NULL;
      numChannels = 0;
   }
}

/**
 * Primary method for derived layer initialization.  This should be called
 * after initialize_base has been called.
 * WARNING, should only called by derived class
 * (initialize is not called by base HyPerLayer, which is an abstract class).
 */
int HyPerLayer::initialize(PVLayerType type)
{
   int status = PV_SUCCESS;
   float time = 0.0f;

   // IMPORTANT:
   //   - all derived classes should make sure that HyPerLayer::initialize is called
   //
   clayer->layerType = type;
   parent->addLayer(this);

   bool restart_flag = parent->parameters()->value(name, "restart", 0.0f) != 0.0f;
   initializeV(restart_flag);
   if( restart_flag ) {
      readState(&time);
   }

   return status;
}

int HyPerLayer::initialize_base(const char * name, HyPerCol * hc, int numChannels)
{
   // This should have only what's absolutely essential to all HyPerLayers, since nothing in it can be overridden
   PVLayerLoc layerLoc;

   // name should be initialized first as other methods may use it
   this->name = strdup(name);
   setParent(hc);

   this->probes = NULL;
   this->ioAppend = 0;
   this->numProbes = 0;

   this->numChannels = numChannels;

#ifdef PV_USE_OPENCL
   this->numEvents = 0;  // reset by derived classes
   this->numWait   = 0;
   this->numKernelArgs = 0;
#endif

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

   hc->setLayerLoc(&layerLoc, nxScale, nyScale, margin, numFeatures);

   float xScalef = -log2f( (float) nxScale);
   float yScalef = -log2f( (float) nyScale);

   int xScale = (int) nearbyintf(xScalef);
   int yScale = (int) nearbyintf(yScalef);

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

   mirrorBCflag = (bool) params->value(name, "mirrorBCflag", 0);

   clayer = pvlayer_new(layerLoc, xScale, yScale, numChannels);
   // Initializing of V moved into HyPerLayer::initialize() by means of method initializeV(), where it can be overridden.
   clayer->layerType = TypeGeneric;

   // allocate storage for the input conductance arrays
   //
   GSyn = NULL;
   if (numChannels > 0) {
      GSyn = (pvdata_t **) malloc(numChannels*sizeof(pvdata_t *));
      assert(GSyn != NULL);

      GSyn[0] = (pvdata_t *) calloc(getNumNeurons()*numChannels, sizeof(pvdata_t));
      assert(GSyn[0] != NULL);

      for (int m = 1; m < numChannels; m++) {
         GSyn[m] = GSyn[0] + m * getNumNeurons();
      }
   }

   // labels are not extended
   labels = (int *) calloc(getNumNeurons(), sizeof(int));
   assert(labels != NULL);

   marginIndices = NULL; //getMarginIndices();  // only store if getMarginIndices() is called

   return 0;
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
   if( parent->includeLayerName() ) {
      snprintf(filename, PV_PATH_MAX, "%s/a%d_%s.pvp", parent->getOutputPath(), clayer->layerId, name);
   } else {
      snprintf(filename, PV_PATH_MAX, "%s/a%d.pvp", parent->getOutputPath(), clayer->layerId);
   }
   clayer->activeFP = pvp_open_write_file(filename, parent->icCommunicator(), append);

   return 0;
}

int HyPerLayer::initializeV(bool restart_flag) {
   float Vrest = parent->parameters()->value(name, "Vrest", V_REST);
   if( !restart_flag ) {
      for (int k = 0; k < this->getNumNeurons(); k++){
         getV()[k] = Vrest;
      }
   }
   // If restart_flag is true, initialize() will set V by calling readState()
   return PV_SUCCESS;
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
   clV        = device->createBuffer(CL_MEM_COPY_HOST_PTR, size,    clayer->V);
   clActivity = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->activity->data);
   clPrevTime = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->prevActivity);

   // defer creation of clParams to derived classes (as it is class specific)

   clGSyn = NULL;
   if (numChannels > 0) {
      clGSyn = (CLBuffer **) malloc(numChannels*sizeof(CLBuffer *));
      assert(clGSyn != NULL);

      for (int m = 0; m < numChannels; m++) {
         clGSyn[m] = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, GSyn[m]);
      }
   }

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

   comm->addPublisher(this, clayer->activity->numItems, MAX_F_DELAY);

   return 0;
}

int HyPerLayer::initFinish()
{
   return 0;
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
   return store->bufferOffset(LOCAL, delay);
}

CLBuffer * HyPerLayer::getLayerDataStoreCLBuffer()
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   return store->getCLBuffer();
}
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
#ifdef OBSOLETE  // Marked obsolete Aug 31, 2011.  Size of the buffer is nx*ny*nf whether buffer is extended or not.
   // If extended is true, there will be space at the end of the buffer.
   // Clear it so that the same output file is always produced from the
   // same value of data.
   while(ii<numItems) buf[ii++] = 0;
#endif OBSOLETE
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

int HyPerLayer::updateState(float time, float dt)
{
   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)

   updateV();
   setActivity();
   resetGSynBuffers();
   updateActiveIndices();

   return 0;
}

int HyPerLayer::updateBorder(float time, float dt)
{
   int status = PV_SUCCESS;

#ifdef PV_USE_OPENCL
#if PV_CL_EVENTS
   // wait for memory to be copied from device
   if (numWait > 0) {
      status |= clWaitForEvents(numWait, evList);
   }
   for (int i = 0; i < numWait; i++) {
      clReleaseEvent(evList[i]);
   }
   numWait = 0;

   status |= clWaitForEvents(1, &evUpdate);
   clReleaseEvent(evUpdate);
#endif
#endif

   return status;
}

int HyPerLayer::updateV() {
   pvdata_t * V = getV();
   pvdata_t * GSynExc = getChannel(CHANNEL_EXC);
   pvdata_t * GSynInh = getChannel(CHANNEL_INH);
   for( int k=0; k<getNumNeurons(); k++ ) {
      V[k] = GSynExc[k] - GSynInh[k];
// functionality of MAX and THRESH moved to ANNLayer
//#undef SET_MAX
//#ifdef SET_MAX
//      V[k] = V[k] > 1.0f ? 1.0f : V[k];
//#endif
//#undef SET_THRESH
//#ifdef SET_THRESH
//      V[k] = V[k] < 0.5f ? 0.0f : V[k];
//#endif
   }
   return PV_SUCCESS;
}

int HyPerLayer::updateActiveIndices(){
   if (!spikingFlag) return PV_SUCCESS;
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



int HyPerLayer::setActivity() {
   const int nx = getLayerLoc()->nx;
   const int ny = getLayerLoc()->ny;
   const int nf = getLayerLoc()->nf;
   const int nb = getLayerLoc()->nb;
   pvdata_t * activity = getCLayer()->activity->data;
   pvdata_t * V = getV();
   for( int k=0; k<getNumExtended(); k++ ) {
      activity[k] = 0; // Would it be faster to only do the margins?
   }
   for( int k=0; k<getNumNeurons(); k++ ) {
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      activity[kex] = V[k];
   }
   return PV_SUCCESS;
}

int HyPerLayer::resetGSynBuffers() {
   int n = getNumNeurons();
   for( int k=0; k<numChannels; k++ ) {
      resetBuffer( getChannel((ChannelType) k), n );
   }
   // resetBuffer( getChannel(CHANNEL_EXC), n );
   // resetBuffer( getChannel(CHANNEL_INH), n );
   return PV_SUCCESS;
}

int HyPerLayer::resetBuffer( pvdata_t * buf, int numItems ) {
   assert(buf);
   for( int k=0; k<numItems; k++ ) buf[k] = 0.0;
   return PV_SUCCESS;
}

int HyPerLayer::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID)
{
   recvsyn_timer->start();

   assert(arborID >= 0);
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   fflush(stdout);
#endif

   for (int kPre = 0; kPre < numExtended; kPre++) {
      float a = activity->data[kPre];

      // Activity < 0 is used by generative models --pete
      if (a == 0.0f) continue;  // TODO - assume activity is sparse so make this common branch

      PVAxonalArbor * arbor = conn->axonalArbor(kPre, arborID);
      PVPatch * GSyn = arbor->data;
      PVPatch * weights = arbor->weights;

      // WARNING - assumes weight and GSyn patches from task same size
      //         - assumes patch stride sf is 1

      int nk  = GSyn->nf * GSyn->nx;
      int ny  = GSyn->ny;
      int sy  = GSyn->sy;       // stride in layer
      int syw = weights->sy;    // stride in patch

      // TODO - unroll
      for (int y = 0; y < ny; y++) {
         (conn->accumulateFunctionPointer)(nk, GSyn->data + y*sy, a, weights->data + y*syw);
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
   return comm->deliver(parent, getLayerId());
}

int HyPerLayer::publish(InterColComm* comm, float time)
{
   if ( useMirrorBCs() ) {
      for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
         mirrorInteriorToBorder(borderId, clayer->activity, clayer->activity);
      }
   }
   comm->publish(this, clayer->activity);
   return 0;
}

int HyPerLayer::waitOnPublish(InterColComm* comm)
{
   // wait for MPI border transfers to complete
   //
   return comm->wait(getLayerId());
}
//
/* Inserts a new probe into an array of LayerProbes.
 *
 *
 *
 */
int HyPerLayer::insertProbe(LayerProbe * p)
{
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

int HyPerLayer::outputState(float time, bool last)
{
   int status = PV_SUCCESS;

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(time, this);
   }


   if (time >= writeTime && writeStep >= 0) {
      writeTime += writeStep;
      if (spikingFlag != 0) {
         status = writeActivitySparse(time);
      }
      else {
         if (writeNonspikingActivity) {
            status = writeActivity(time);
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

int HyPerLayer::readState(float * time)
{
   double dtime;
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   int status = PV_SUCCESS;

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   if( getV() != NULL ) {
      getOutputFilename(path, "V", "_last");
      status = read(path, comm, &dtime, getV(), loc, PV_FLOAT_TYPE, extended, contiguous);
      assert(status == PV_SUCCESS);
   }

   getOutputFilename(path, "labels", "");
   status = read(path, comm, &dtime, (float*)labels, loc, PV_INT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS || status == PV_ERR_FILE_NOT_FOUND);  // not required to exist
   if (status == PV_ERR_FILE_NOT_FOUND) {
      if (labels != NULL) free(labels);
      labels = NULL;
   }

   // TODO - this should be moved to getLayerData but can't yet because publish is call
   // as the first step and publish copies clayer->activity->data into data store.  If
   // clayer->activity is removed then we would read directly into data store.
   extended = true;
   pvdata_t * A = clayer->activity->data;
   getOutputFilename(path, "A", "_last");
   status = read(path, comm, &dtime, A, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   return status;
}

int HyPerLayer::writeState(float time, bool last)
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
      status |= write_pvdata(path, comm, time, getV(), loc, PV_FLOAT_TYPE, extended, contiguous);
   }

   extended = true;
   getOutputFilename(path, "A", last_str);
   status |= write_pvdata(path, comm, time, getLayerData(), loc, PV_FLOAT_TYPE, extended, contiguous);

   return status;
}

int HyPerLayer::writeActivitySparse(float time)
{
   // calculate active indices  -- Moved to HyPerLayer method
   //
//   int numActive = 0;
//   PVLayerLoc & loc = clayer->loc;
//   pvdata_t * activity = clayer->activity->data;
//
//   for (int k = 0; k < getNumNeurons(); k++) {
//      const int kex = kIndexExtended(k, loc.nx, loc.ny, loc.nf, loc.nb);
//      if (activity[kex] > 0.0) {
//         clayer->activeIndices[numActive++] = globalIndexFromLocal(k, loc);
//      }
//   }
//   clayer->numActive = numActive;

   return PV::writeActivitySparse(clayer->activeFP, parent->icCommunicator(), time, clayer);
}

// write non-spiking activity
int HyPerLayer::writeActivity(float time)
{
   // currently numActive only used by writeActivitySparse
   //
   clayer->numActive = 0;

   return PV::writeActivity(clayer->activeFP, parent->icCommunicator(), time, clayer);
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

