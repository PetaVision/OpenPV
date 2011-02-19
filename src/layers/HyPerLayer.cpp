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
#include "../io/fileio.hpp"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

#ifdef OBSOLETE
HyPerLayerParams defaultParams =
{
    V_REST, V_EXC, V_INH, V_INHB,            // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,           // tau (ms)
    250, NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    250, NOISE_AMP*1.0,
    250, NOISE_AMP*1.0                       // noise (G)
};
#endif

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

   if (numChannels > 0) {
      // potentials allocated contiguously so this frees all
      free(phi[0]);
   }
}

/**
 * Primary method for derived layer initialization.  This should be called
 * after initialize_base has been called.  WARNING, very little should be
 * done here as it should be done in derived class.
 */
int HyPerLayer::initialize(PVLayerType type)
{
   int status = 0;
   float time = 0.0f;

   // IMPORTANT:
   //   - these two statements should be done in all derived classes
   //
   clayer->layerType = type;
   parent->addLayer(this);

   if (parent->parameters()->value(name, "restart", 0.0f) != 0.0f) {
      readState(&time);
   }

   return status;
}

int HyPerLayer::initialize_base(const char * name, HyPerCol * hc, int numChannels)
{
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

   mirrorBCflag = (bool) params->value(name, "mirrorBCflag", 0);

   clayer = pvlayer_new(layerLoc, xScale, yScale, numChannels);
   float Vrest = params->value(name, "Vrest", V_REST);
   for (int k = 0; k < this->getNumNeurons(); k++){
      getV()[k] = Vrest;
   }

   clayer->layerType = TypeGeneric;

   for (int m = 1; m < MAX_CHANNELS; m++) {
      phi[m] = NULL;
   }
   if (numChannels > 0) {
      phi[0] = (pvdata_t *) calloc(getNumNeurons()*numChannels, sizeof(pvdata_t));
      assert(phi[0] != NULL);

      for (int m = 1; m < numChannels; m++) {
         phi[m] = phi[0] + m * getNumNeurons();
      }
   }

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

   sprintf(filename, "%s/a%d.pvp", OUTPUT_PATH, clayer->layerId);
   clayer->activeFP = pvp_open_write_file(filename, parent->icCommunicator(), append);

   return 0;
}

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
const pvdata_t * HyPerLayer::getLayerData()
{
   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
   return (pvdata_t *) store->buffer(LOCAL);
}


// deprecated?
/**
 * returns the number of neurons in the layer or border region
 * @param borderId the id of the border region (0 for interior/self)
 **/
int HyPerLayer::numberOfNeurons(int borderId)
{
   int numNeurons = 0;
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
   }

   return -1;
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

int HyPerLayer::copyFromBuffer(const pvdata_t * buf, pvdata_t * data,
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
            data[iex*sx + jex*sy + f*sf] = scale * buf[ii++];
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
   resetPhiBuffers();

   return 0;
}

int HyPerLayer::updateBorder(float time, float dt)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL
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

   return status;
}

int HyPerLayer::updateV() {
   pvdata_t * V = getV();
   pvdata_t * phiExc = getChannel(CHANNEL_EXC);
   pvdata_t * phiInh = getChannel(CHANNEL_INH);
   for( int k=0; k<getNumNeurons(); k++ ) {
      V[k] = phiExc[k] - phiInh[k];
#undef SET_MAX
#ifdef SET_MAX
      V[k] = V[k] > 1.0f ? 1.0f : V[k];
#endif
#undef SET_THRESH
#ifdef SET_THRESH
      V[k] = V[k] < 0.5f ? 0.0f : V[k];
#endif
   }
   return EXIT_SUCCESS;
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
   return EXIT_SUCCESS;
}

int HyPerLayer::resetPhiBuffers() {
   int n = getNumNeurons();
   resetBuffer( getChannel(CHANNEL_EXC), n );
   resetBuffer( getChannel(CHANNEL_INH), n );
   return EXIT_SUCCESS;
}

int HyPerLayer::resetBuffer( pvdata_t * buf, int numItems ) {
   for( int k=0; k<numItems; k++ ) buf[k] = 0.0;
   return EXIT_SUCCESS;
}

int HyPerLayer::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   recvsyn_timer->start();

   assert(neighbor >= 0);
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

      PVAxonalArbor * arbor = conn->axonalArbor(kPre, neighbor);
      PVPatch * phi = arbor->data;
      PVPatch * weights = arbor->weights;

      // WARNING - assumes weight and phi patches from task same size
      //         - assumes patch stride sf is 1

      int nk  = phi->nf * phi->nx;
      int ny  = phi->ny;
      int sy  = phi->sy;        // stride in layer
      int syw = weights->sy;    // stride in patch

      // TODO - unroll
      for (int y = 0; y < ny; y++) {
         pvpatch_accumulate(nk, phi->data + y*sy, a, weights->data + y*syw);
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
   int status = 0;
#ifdef OBSOLETE
   char path[PV_PATH_MAX];

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->numFeatures;

   const int nxex = clayer->loc.nx + 2*clayer->loc.nb;
   const int nyex = clayer->loc.ny + 2*clayer->loc.nb;
#endif

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(time, this);
   }

   PVParams * params = parent->parameters();
   int spikingFlag = (int) params->value(name, "spikingFlag", 1);

#undef WRITE_NONSPIKING_ACTIVITY
#ifdef WRITE_NONSPIKING_ACTIVITY
   float defaultWriteNonspikingActivity = 1.0;
#else
   float defaultWriteNonspikingActivity = 0.0;
#endif

   if (spikingFlag != 0) {
      status = writeActivitySparse(time);
   }
   else {
      int writeNonspikingActivity = (int) params->value(name, "writeNonspikingActivity",
            defaultWriteNonspikingActivity);
      if (writeNonspikingActivity) {
         status = writeActivity(time);
      }
   }

   if (time >= writeTime) {
      writeTime += writeStep;
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
   snprintf(buf, PV_PATH_MAX-1, "%s%s_%s%s.pvp", OUTPUT_PATH, getName(), dataName, term);
   return buf;
}

int HyPerLayer::readState(float * time)
{
   double dtime;
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   int status = 0;

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   getOutputFilename(path, "V", "_last");
   status |= read(path, comm, &dtime, clayer->V, loc, PV_FLOAT_TYPE, extended, contiguous);

   // TODO - this should be moved to getLayerData but can't yet because publish is call
   // as the first step and publish copies clayer->activity->data into data store.  If
   // clayer->activity is removed then we would read directly into data store.
   extended = true;
   pvdata_t * A = clayer->activity->data;
   getOutputFilename(path, "A", "_last");
   status |= read(path, comm, &dtime, A, loc, PV_FLOAT_TYPE, extended, contiguous);

   return status;
}

int HyPerLayer::writeState(float time, bool last)
{
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   int status = 0;

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   const char * last_str = (last) ? "_last" : "";

   getOutputFilename(path, "V", last_str);
   status |= write(path, comm, time, clayer->V, loc, PV_FLOAT_TYPE, extended, contiguous);

   extended = true;
   getOutputFilename(path, "A", last_str);
   status |= write(path, comm, time, getLayerData(), loc, PV_FLOAT_TYPE, extended, contiguous);

   return status;
}

int HyPerLayer::writeActivitySparse(float time)
{
   // calculate active indices
   //
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

#ifdef OBSOLETE // (marked obsolete Jan 24, 2011)
// modified to enable writing of non-spiking activity as well
// use writeActivitySparse for efficient disk storage of sparse spiking activity
int HyPerLayer::writeActivity(const char * filename, float time)
{
   int status = 0;
   PVLayerLoc * loc = &clayer->loc;

   const int n = loc->nx * loc->ny * loc->nf;
   pvdata_t * buf = new pvdata_t[n];
   assert(buf != NULL);

   const bool extended = true;
   status = copyToBuffer(buf, getLayerData(), loc, extended, 1.0);

   // gather the local portions and write the image
   status = gatherImageFile(filename, parent->icCommunicator(), loc, buf);

   delete buf;

   return status;
}
#endif // OBSOLETE

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

} // end of PV namespace

