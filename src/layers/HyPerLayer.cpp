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

#ifdef __cplusplus
extern "C" {
#endif

static int    pvcube_init(PVLayerCube * cube, PVLayerLoc * loc, int numItems);
static size_t pvcube_size(int numItems);

#ifdef __cplusplus
}
#endif

namespace PV {

HyPerLayerParams defaultParams =
{
    V_REST, V_EXC, V_INH, V_INHB,            // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,           // tau (ms)
    250, NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    250, NOISE_AMP*1.0,
    250, NOISE_AMP*1.0                       // noise (G)
};

///////
// This constructor is protected so that only derived classes can call it.
// It should be called as the normal method of object construction by
// derived classes.  It should NOT call any virtual methods
HyPerLayer::HyPerLayer(const char* name, HyPerCol * hc)
{
   initialize_base(name, hc);
}

HyPerLayer::~HyPerLayer()
{
   if (clayer != NULL) {
      // pvlayer_finalize will free clayer
      pvlayer_finalize(clayer);
      clayer = NULL;
   }
   free(name);
}

int HyPerLayer::initialize(PVLayerType type)
{
   float time = 0.0;

   clayer->layerType = type;
   parent->addLayer(this);

   if (parent->parameters()->value(name, "restart", 0) != 0) {
      readState(name, &time);
   }
   writeTime = parent->simulationTime();

   return 0;
}

int HyPerLayer::initialize_base(const char * name, HyPerCol * hc)
{
   // name should be initialize first as other methods may use it
   this->name = strdup(name);
   setParent(hc);

   this->probes = NULL;
   this->ioAppend = 0;
   this->numProbes = 0;

   PVLayerLoc imageLoc = parent->getImageLoc();

   PVParams * params = parent->parameters();

   int nx = (int) params->value(name, "nx", imageLoc.nx);
   int ny = (int) params->value(name, "ny", imageLoc.ny);

   int numFeatures = (int) params->value(name, "nf", 1);
   int nBorder     = (int) params->value(name, "marginWidth", 0);

   // let nxScale, nyScale supersede nx, ny
   if (params->present(name, "nxScale")) {
      float nxScale = params->value(name, "nxScale");
      nx = (int) nearbyintf( nxScale * imageLoc.nx );
   }
   if (params->present(name, "nyScale")) {
      float nyScale = params->value(name, "nyScale");
      ny = (int) nearbyintf( nyScale * imageLoc.ny );
   }

   float xScalef = log2f(parent->width() / nx);
   float yScalef = log2f(parent->height() / ny);

   int xScale = (int) nearbyintf(xScalef);
   int yScale = (int) nearbyintf(yScalef);

   clayer = pvlayer_new(xScale, yScale, nx, ny, numFeatures, nBorder);
   clayer->layerType = TypeGeneric;

   writeStep = params->value(name, "writeStep", parent->getDeltaTime());

   return 0;
}

int HyPerLayer::initBorder(PVLayerCube * border, int borderId)
{
   // TODO - this uses clayer nxGlobal and nyGlobal
   // TODO - is this correct, kx0 or ky0 < 0
   // TODO - does global patch need to expand to take into account border regions (probably)

   PVLayerLoc loc = clayer->loc;

   const int nxBorder = loc.nPad;
   const int nyBorder = loc.nPad;

   switch (borderId) {
   case NORTHWEST:
      loc.nx = nxBorder;
      loc.ny = nyBorder;
      loc.kx0 = clayer->loc.kx0 - nxBorder;
      loc.ky0 = clayer->loc.ky0 - nyBorder;
      break;
   case NORTH:
      loc.ny = nyBorder;
      loc.ky0 = clayer->loc.ky0 - nyBorder;
      break;
   case NORTHEAST:
      loc.nx = nxBorder;
      loc.ny = nyBorder;
      loc.kx0 = clayer->loc.kx0 + clayer->loc.nx;
      loc.ky0 = clayer->loc.ky0 - nyBorder;
      break;
   case WEST:
      loc.nx = nxBorder;
      loc.kx0 = clayer->loc.kx0 - nxBorder;
      break;
   case EAST:
      loc.nx = nxBorder;
      loc.kx0 = clayer->loc.kx0 + clayer->loc.nx;
      break;
   case SOUTHWEST:
      loc.nx = nxBorder;
      loc.ny = nyBorder;
      loc.kx0 = clayer->loc.kx0 - nxBorder;
      loc.ky0 = clayer->loc.ky0 + clayer->loc.ny;
      break;
   case SOUTH:
      loc.ny = nyBorder;
      loc.ky0 = clayer->loc.ky0 + clayer->loc.ny;
      break;
   case SOUTHEAST:
      loc.nx = nxBorder;
      loc.ny = nyBorder;
      loc.kx0 = clayer->loc.kx0 + clayer->loc.nx;
      loc.ky0 = clayer->loc.ky0 + clayer->loc.ny;
      break;
   default:
      fprintf(stderr, "ERROR:HyPerLayer:initBorder: bad border index %d\n", borderId);
   }

   pvcube_init(border, &loc, loc.nx * loc.ny * clayer->numFeatures);

   return 0;
}

int HyPerLayer::initGlobal(int colId, int colRow, int colCol, int nRows, int nCols)
{
   char filename[PV_PATH_MAX];
   bool append = false;

   sprintf(filename, "%s/a%d.pvp", OUTPUT_PATH, clayer->layerId);
   clayer->activeFP = pvp_open_write_file(filename, parent->icCommunicator(), append);

   return pvlayer_initGlobal(clayer, colId, colRow, colCol, nRows, nCols);
}

int HyPerLayer::columnWillAddLayer(InterColComm * comm, int layerId)
{
   setLayerId(layerId);

   // complete initialization now that we have a parent and a communicator
   // WARNING - this must be done before addPublisher is called
   int id = parent->columnId();
   initGlobal(id, comm->commRow(), comm->commColumn(),
                  comm->numCommRows(), comm->numCommColumns());

   comm->addPublisher(this, clayer->activity->numItems, MAX_F_DELAY);

   return 0;
}

int HyPerLayer::initFinish()
{
   return pvlayer_initFinish(clayer);
}

/**
 * returns the number of neurons in the layer or border region
 * @param borderId the id of the border region (0 for interior/self)
 **/
int HyPerLayer::numberOfNeurons(int borderId)
{
   int numNeurons;
   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->numFeatures;
   const int nxBorder = clayer->loc.nPad;
   const int nyBorder = clayer->loc.nPad;

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
 */
int HyPerLayer::copyToBorder(int whichBorder, PVLayerCube * cube, PVLayerCube * border)
{
   switch (whichBorder) {
   case NORTHWEST:
      return copyToNorthWest(border, cube);
   case NORTH:
      return copyToNorth(border, cube);
   case NORTHEAST:
      return copyToNorthEast(border, cube);
   case WEST:
      return copyToWest(border, cube);
   case EAST:
      return copyToEast(border, cube);
   case SOUTHWEST:
      return copyToSouthWest(border, cube);
   case SOUTH:
      return copyToSouth(border, cube);
   case SOUTHEAST:
      return copyToSouthEast(border, cube);
   default:
      fprintf(stderr, "ERROR:HyPerLayer:copyToBorder: bad border index %d\n", whichBorder);
   }

   return -1;
}

int HyPerLayer::copyToInteriorBuffer(unsigned char * buf)
{
   return HyPerLayer::copyToBuffer(buf, getLayerData(), getLayerLoc(), isExtended(), 255.0);
}

int HyPerLayer::copyToBuffer(unsigned char * buf, const pvdata_t * data,
                             const PVLayerLoc * loc, bool extended, float scale)
{
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nBands;

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->nPad;
      nyBorder = loc->nPad;
   }

   const int sf = 1;
   const int sx = nf * sf;
   const int sy = sx * (nx + 2*nxBorder);

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
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nBands;

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->nPad;
      nyBorder = loc->nPad;
   }

   const int sf = 1;
   const int sx = nf * sf;
   const int sy = sx * (nx + 2*nxBorder);

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
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nBands;

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->nPad;
      nyBorder = loc->nPad;
   }

   const int sf = 1;
   const int sx = nf * sf;
   const int sy = sx * (nx + 2*nxBorder);

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
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nBands;

   int nxBorder = 0;
   int nyBorder = 0;

   if (extended) {
      nxBorder = loc->nPad;
      nyBorder = loc->nPad;
   }

   const int sf = 1;
   const int sx = nf * sf;
   const int sy = sx * (nx + 2*nxBorder);

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

int HyPerLayer::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   assert(neighbor == 0);
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   fflush(stdout);
#endif

   for (int kPre = 0; kPre < numExtended; kPre++) {
      float a = activity->data[kPre];
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

   return 0;
}

int HyPerLayer::reconstruct(HyPerConn * conn, PVLayerCube * cube)
{
   // TODO - implement
   printf("[%d]: HyPerLayer::reconstruct: to layer %d from %d\n",
          clayer->columnId, clayer->layerId, conn->preSynapticLayer()->clayer->layerId);
   return 0;
}

int HyPerLayer::publish(InterColComm* comm, float time)
{
   comm->publish(this, clayer->activity);
   return 0;
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
//   char path[PV_PATH_MAX];
   int status = 0;

//   const int nx = clayer->loc.nx;
//   const int ny = clayer->loc.ny;
//   const int nf = clayer->numFeatures;

//   const int nxex = clayer->loc.nx + 2*clayer->loc.nPad;
//   const int nyex = clayer->loc.ny + 2*clayer->loc.nPad;

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(time, this);
   }

   // always write activity in sparse format
   status = writeActivitySparse(time);

   if (time >= writeTime) {
      writeTime += writeStep;

      // should use a probe to get runtime information
      //

      //snprintf(path, PV_PATH_MAX-1, "%s%s_A.gif", OUTPUT_PATH, name);
      //writeActivity(path, time);

      // this output format is decrecated as it doesn't use MPI
      //
      //sprintf(path, "A%1.1d", clayer->layerId);
      //pv_dump(path, ioAppend, clayer->activity->data, nxex, nyex, nf);
      //sprintf(path, "V%1.1d", clayer->layerId);
      //pv_dump(path, ioAppend, clayer->V, nx, ny, nf);
      // append to dump file after original open
      //this->ioAppend = 1;
   }

   return status;
}

int HyPerLayer::readState(const char * name, float * time)
{
   int status = 0;
   char path[PV_PATH_MAX];

   double dtime;

   bool contiguous = false;
   bool extended   = false;

   Communicator * comm = parent->icCommunicator();

   const char * last = "_last";
   const char * name_str = (name_str != NULL  ) ?  name   : "";

   pvdata_t * G_E  = clayer->G_E;
   pvdata_t * G_I  = clayer->G_I;

   pvdata_t * V   = clayer->V;
   pvdata_t * Vth = clayer->Vth;

   pvdata_t * A = clayer->activity->data;

   PVLayerLoc * loc = & clayer->loc;

   snprintf(path, PV_PATH_MAX-1, "%s%s_GE%s.pvp", OUTPUT_PATH, name_str, last);
   status = read(path, comm, &dtime, G_E, loc, PV_FLOAT_TYPE, extended, contiguous);

   snprintf(path, PV_PATH_MAX-1, "%s%s_GI%s.pvp", OUTPUT_PATH, name_str, last);
   status = read(path, comm, &dtime, G_I, loc, PV_FLOAT_TYPE, extended, contiguous);

   snprintf(path, PV_PATH_MAX-1, "%s%s_V%s.pvp", OUTPUT_PATH, name_str, last);
   status = read(path, comm, &dtime, V, loc, PV_FLOAT_TYPE, extended, contiguous);

   snprintf(path, PV_PATH_MAX-1, "%s%s_Vth%s.pvp", OUTPUT_PATH, name_str, last);
   status = read(path, comm, &dtime, Vth, loc, PV_FLOAT_TYPE, extended, contiguous);

   extended = true;
   snprintf(path, PV_PATH_MAX-1, "%s%s_A%s.pvp", OUTPUT_PATH, name_str, last);
   status = read(path, comm, &dtime, A, loc, PV_FLOAT_TYPE, extended, contiguous);

   return status;
}

int HyPerLayer::writeState(const char * name, float time, bool last)
{
   int status = 0;
   char path[PV_PATH_MAX];

   bool contiguous = false;
   bool extended   = false;

   Communicator * comm = parent->icCommunicator();

   const char * last_str = (last) ? "_last" : "";
   const char * name_str = (name_str != NULL  ) ?  name   : "";

   pvdata_t * G_E  = clayer->G_E;
   pvdata_t * G_I  = clayer->G_I;

   pvdata_t * V   = clayer->V;
   pvdata_t * Vth = clayer->Vth;

   pvdata_t * A = clayer->activity->data;

   PVLayerLoc * loc = & clayer->loc;

   snprintf(path, PV_PATH_MAX-1, "%s%s_GE%s.pvp", OUTPUT_PATH, name_str, last_str);
   status = write(path, comm, time, G_E, loc, PV_FLOAT_TYPE, extended, contiguous);

   snprintf(path, PV_PATH_MAX-1, "%s%s_GI%s.pvp", OUTPUT_PATH, name_str, last_str);
   status = write(path, comm, time, G_I, loc, PV_FLOAT_TYPE, extended, contiguous);

   snprintf(path, PV_PATH_MAX-1, "%s%s_V%s.pvp", OUTPUT_PATH, name_str, last_str);
   status = write(path, comm, time, V, loc, PV_FLOAT_TYPE, extended, contiguous);

   snprintf(path, PV_PATH_MAX-1, "%s%s_Vth%s.pvp", OUTPUT_PATH, name_str, last_str);
   status = write(path, comm, time, Vth, loc, PV_FLOAT_TYPE, extended, contiguous);

   extended = true;
   snprintf(path, PV_PATH_MAX-1, "%s%s_A%s.pvp", OUTPUT_PATH, name_str, last_str);
   status = write(path, comm, time, A, loc, PV_FLOAT_TYPE, extended, contiguous);

   return 0;
}

int HyPerLayer::writeActivitySparse(float time)
{
   return PV::writeActivitySparse(clayer->activeFP, parent->icCommunicator(), time, clayer);
}

int HyPerLayer::writeActivity(const char * filename, float time)
{
   int status = 0;
   PVLayerLoc * loc = &clayer->loc;

   const int n = loc->nx * loc->ny * loc->nBands;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   const bool extended = true;
   status = copyToBuffer(buf, clayer->activity->data, loc, extended, 255);

   // gather the local portions and write the image
   status = gatherImageFile(filename, parent->icCommunicator(), loc, buf);

   delete buf;

   return status;
}

int HyPerLayer::setParams(int numParams, size_t sizeParams, float * params)
{
   return pvlayer_setParams(clayer, numParams, sizeParams, params);
}

int HyPerLayer::setFuncs(void * initFunc, void * updateFunc)
{
   return pvlayer_setFuncs(clayer, initFunc, updateFunc);
}

int HyPerLayer::getParams(int * numParams, float ** params)
{
   return pvlayer_getParams(clayer, numParams, params);
}

#ifndef FEATURES_LAST
static int copyNS(pvdata_t * dest, pvdata_t * src, int nk)
{
   for (int k = 0; k < nk; k++) {
      dest[k] = src[k];  // TODO - use memcpy?
   }
   return 0;
}

static int copyEW(pvdata_t * dest, pvdata_t * src, int nf, int ny, int syDst, int sySrc)
{
   pvdata_t * to   = dest;
   pvdata_t * from = src;

   for (int j = 0; j < ny; j++) {
      for (int f = 0; f < nf; f++) {
         to[f] = from[f];
      }
      to   += syDst;
      from += sySrc;
   }
   return 0;
}

static int copyTopCorner(pvdata_t * dest, pvdata_t * src, int nf, int ny, int syDst, int sySrc)
{
   pvdata_t * to   = dest;
   pvdata_t * from = src;

   for (int j = 0; j < ny; j++) {
      for (int f = 0; f < nf; f++) {
         to[f] = from[f];
      }
      to   -= syDst;
      from += sySrc;
   }
   return 0;
}

static int copyBottomCorner(pvdata_t * dest, pvdata_t * src, int nf, int ny, int syDst, int sySrc)
{
   pvdata_t * to   = dest;
   pvdata_t * from = src;

   for (int j = 0; j < ny; j++) {
      for (int f = 0; f < nf; f++) {
         to[f] = from[f];
      }
      to   += syDst;
      from -= sySrc;
   }
   return 0;
}

int HyPerLayer::copyToNorthWest(PVLayerCube * dest, PVLayerCube * src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src-> data;
   pvdata_t * dst0 = dest->data + (nx-1)*nf + (ny-1)*syDst;

   for (int i = 0; i < nx; i++) {
      pvdata_t * to   = dst0 - i*nf;
      pvdata_t * from = src0 + i*nf;
      copyTopCorner(to, from, nf, ny, syDst, sySrc);
   }
   return 0;
}

int HyPerLayer::copyToNorth(PVLayerCube * dest, PVLayerCube * src)
{
   int ny = dest->loc.ny;
   int sy = clayer->numFeatures * dest->loc.nx;

   pvdata_t * src0 = src-> data;
   pvdata_t * dst0 = dest->data + (ny-1)*sy;

   for (int j = 0; j < ny; j++) {
      pvdata_t * to   = dst0 - j*sy;
      pvdata_t * from = src0 + j*sy;
      copyNS(to, from, sy);
   }
   return 0;
}

int HyPerLayer::copyToNorthEast(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src ->data + (src->loc.nx - 1)*nf;
   pvdata_t * dst0 = dest->data + (ny-1)*syDst;

   for (int i = 0; i < nx; i++) {
      pvdata_t * to   = dst0 + i*nf;
      pvdata_t * from = src0 - i*nf;
      copyTopCorner(to, from, nf, ny, syDst, sySrc);
   }
   return 0;
}

int HyPerLayer::copyToWest(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src ->data;
   pvdata_t * dst0 = dest->data + (nx-1)*nf;

   for (int i = 0; i < nx; i++) {
      pvdata_t * to   = dst0 - i*nf;
      pvdata_t * from = src0 + i*nf;
      copyEW(to, from, nf, ny, syDst, sySrc);
   }
   return 0;
}

int HyPerLayer::copyToEast(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src ->data + (src->loc.nx - 1)*nf;
   pvdata_t * dst0 = dest->data;

   for (int i = 0; i < nx; i++) {
      pvdata_t * to   = dst0 + i*nf;
      pvdata_t * from = src0 - i*nf;
      copyEW(to, from, nf, ny, syDst, sySrc);
   }
   return 0;
}

int HyPerLayer::copyToSouthWest(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src-> data + (src->loc.ny - 1)*sySrc;
   pvdata_t * dst0 = dest->data + (nx - 1)*nf;

   for (int i = 0; i < nx; i++) {
      pvdata_t * to   = dst0 - i*nf;
      pvdata_t * from = src0 + i*nf;
      copyBottomCorner(to, from, nf, ny, syDst, sySrc);
   }
   return 0;
}

int HyPerLayer::copyToSouth(PVLayerCube* dest, PVLayerCube* src)
{
   int ny = dest->loc.ny;
   int sy = clayer->numFeatures * dest->loc.nx;

   pvdata_t * src0 = src ->data + (src->loc.ny - 1)*sy;
   pvdata_t * dst0 = dest->data;

   for (int j = 0; j < ny; j++) {
      pvdata_t * to   = dst0 + j*sy;
      pvdata_t * from = src0 - j*sy;
      copyNS(to, from, sy);
   }
   return 0;
}

int HyPerLayer::copyToSouthEast(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src-> data + (src->loc.nx - 1)*nf
                                + (src->loc.ny - 1)*sySrc;
   pvdata_t * dst0 = dest->data;

   for (int i = 0; i < nx; i++) {
      pvdata_t * to   = dst0 + i*nf;
      pvdata_t * from = src0 - i*nf;
      copyBottomCorner(to, from, nf, ny, syDst, sySrc);
   }
   return 0;
}

#else // end features first section
static int copyNS(pvdata_t * dest, pvdata_t * src, int nx, int ny, int stride)
{
   pvdata_t * to   = dest;
   pvdata_t * from = src + (ny-1)*stride;

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         to[i] = from[i];  // TODO - use memcpy?
      }
      to   += stride;
      from -= stride;
   }
   return 0;
}

static int copyEW(pvdata_t * dest, pvdata_t * src, int nx, int ny, int nxSrc)
{
   pvdata_t * to   = dest;
   pvdata_t * from = src;

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         to[i] = from[nx-1-i];
      }
      to   += nx;
      from += nxSrc;
   }
   return 0;
}

static int copyCorner(pvdata_t * dest, pvdata_t * src, int nx, int ny, int nxSrc)
{
   pvdata_t * to   = dest;
   pvdata_t * from = src + (ny-1)*nxSrc;

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         to[i] = from[nx-1-i];
      }
      to   += nx;
      from -= nxSrc;
   }
   return 0;
}

int HyPerLayer::copyToNorthWest(PVLayerCube * dest, PVLayerCube * src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = f * src->loc.nx * src->loc.ny;
      copyCorner(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}

int HyPerLayer::copyToNorth(PVLayerCube * dest, PVLayerCube * src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = f * src->loc.nx * src->loc.ny;
      copyNS(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}

int HyPerLayer::copyToNorthEast(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int x = src->loc.nx - nx;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = x + f * src->loc.nx * src->loc.ny;
      copyCorner(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}

int HyPerLayer::copyToWest(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = f * src->loc.nx * src->loc.ny;
      copyEW(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}

int HyPerLayer::copyToEast(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int x = src->loc.nx - nx;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = x + f * src->loc.nx * src->loc.ny;
      copyEW(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}

int HyPerLayer::copyToSouthWest(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int off = (src->loc.ny - ny) * src->loc.nx;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = off + f * src->loc.nx * src->loc.ny;
      copyCorner(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}

int HyPerLayer::copyToSouth(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int off = (src->loc.ny - ny) * src->loc.nx;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = off + f * src->loc.nx * src->loc.ny;
      copyNS(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}

int HyPerLayer::copyToSouthEast(PVLayerCube* dest, PVLayerCube* src)
{
   int nx = dest->loc.nx;
   int ny = dest->loc.ny;
   int x  = src->loc.nx - nx;
   int y  = src->loc.ny - ny;
   int off = x + y * src->loc.nx;

   for (int f = 0; f < clayer->numFeatures; f++) {
      int dOff = f * nx * ny;
      int sOff = off + f * src->loc.nx * src->loc.ny;
      copyCorner(dest->data + dOff, src->data + sOff, nx, ny, src->loc.nx);
   }
   return 0;
}
#endif // end features last section

} // End of PV namespace

#ifdef __cplusplus
extern "C" {
#endif

PVPatch * pvpatch_new(int nx, int ny, int nf)
{
   int sf = 1;
   int sx = nf;
   int sy = sx * nx;

   PVPatch * p = (PVPatch *) malloc(sizeof(PVPatch));
   assert(p != NULL);

   pvdata_t * data = NULL;

   pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}

int pvpatch_delete(PVPatch* p)
{
   free(p);
   return 0;
}

PVPatch * pvpatch_inplace_new(int nx, int ny, int nf)
{
   int sf = 1;
   int sx = nf;
   int sy = sx * nx;

   size_t dataSize = nx * ny * nf * sizeof(float);
   PVPatch * p = (PVPatch *) calloc(sizeof(PVPatch) + dataSize, sizeof(char));
   assert(p != NULL);

   pvdata_t * data = (pvdata_t *) ((char*) p + sizeof(PVPatch));

   pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}

int pvpatch_inplace_delete(PVPatch* p)
{
   free(p);
   return 0;
}

// TODO - make this inline (gcc does it automatically)?
#ifdef REMOVE
static void pvpatch_accumulate_old(PVPatch * phi, float a, PVPatch * weight)
{
   float x, y, f;
   const int nx = phi->nx;
   const int ny = phi->ny;
   const int nf = phi->nf;
   const int sy = phi->sy;
   const int sf = phi->sf;

   // assume unit stride for w (densely packed)
   pvdata_t * w = weight->data;

   for (f = 0; f < nf; f++) {
      for (y = 0; y < ny; y++) {
         pvdata_t * v = phi->data + (int)(y*sy) + (int) (f*sf);

         // there will be at least 4
         v[0] += a * w[0];
         v[1] += a * w[1];
         v[2] += a * w[2];
         v[3] += a * w[3];
         w += 4;

         // do remainder
         for (x = 4; x < nx; x++) {
            *v++ += a * (*w++);
         }
      }
   }
}

/**
 * Return the _global_ (non-extended) leading index in a direction of a patch in the post layer
  * @kPre is the _global_ pre-synaptic index in a direction
 * @k0Post is the index offset in the post layer
 * @scale is the difference in size scale (2^scale) between post and pre layers
 * @nPatch is the size of patch in a given direction
 * @nLocal is the local size of layer in a direction
 */
float pvlayer_patchHead(int kxPre, float kxPost0Left, int xScale, int nPatch)
{
   float shift = 0;
   if (nPatch % 2 == 0) {
      // if even, can't shift evenly (at least for scale < 0)
      // the later choice alternates direction so not always to left
      shift = (xScale < 0) ? 0 : kxPre % 2;
   }
   shift -= (int) (0.5 * nPatch);
   return kxPost0Left + shift + nearby_neighbor(kxPre, xScale);

#ifdef SHIFTED_CENTERS
   // this works better? for nPatch even, not so well for odd
   if (xScale == 0 && (nPatch % 2) == 1) {
      return kxPost0Left + kxPre + 0.5*(1 - nPatch);
   }
   else {
      float a = powf(2.0f,-1.0f*xScale);
      return floorf((kxPost0Left + a*kxPre) + 0.5*(1 - nPatch));
      //      return floorf((kxPost0Left + a*kxPre) + (1.5f - 0.5f*nPatch));
   }
#endif
}
#endif // REMOVE


static size_t pvcube_size(int numItems)
{
   size_t size = LAYER_CUBE_HEADER_SIZE;
   assert(size == EXPECTED_CUBE_HEADER_SIZE); // depends on PV_ARCH_64 setting
   return size + numItems*sizeof(float);
}

static int pvcube_init(PVLayerCube * cube, PVLayerLoc * loc, int numItems)
{
   cube->size = pvcube_size(numItems);
   cube->numItems = numItems;
   cube->loc = *loc;
   pvcube_setAddr(cube);
   return 0;
}

PVLayerCube* pvcube_new(PVLayerLoc * loc, int numItems)
{
   PVLayerCube* cube = (PVLayerCube*) calloc(pvcube_size(numItems), sizeof(char));
   assert(cube !=NULL);
   pvcube_init(cube, loc, numItems);
   return cube;
}

int pvcube_delete(PVLayerCube * cube)
{
   free(cube);
   return 0;
}

int pvcube_setAddr(PVLayerCube * cube)
{
   cube->data = (pvdata_t *) ((char*) cube + LAYER_CUBE_HEADER_SIZE);
   return 0;
}

#ifdef __cplusplus
}
#endif

