/*
 * HyPerLayer.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "HyPerLayer.hpp"
#include "../include/pv_common.h"
#include "../columns/HyPerCol.hpp"
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

HyPerLayer::HyPerLayer()
{
   this->probes = NULL;
   this->ioAppend = 0;
   this->numProbes = 0;
}

HyPerLayer::HyPerLayer(const char* name, HyPerCol * hc)
{
   setParent(hc);
   init(name, TypeGeneric);
   hc->addLayer(this);
}

HyPerLayer::~HyPerLayer()
{
   if (clayer != NULL) {
      // pvlayer_finalize will free clayer
      pvlayer_finalize(clayer);
      clayer = NULL;
   }
}

int HyPerLayer::init(const char * name, PVLayerType type)
{
   this->probes = NULL;
   this->ioAppend = 0;
   this->outputOnPublish = 1;
   this->numProbes = 0;

   PVParams * params = parent->parameters();

   int nBorder = 0;

   float nx = params->value(name, "nx");
   float ny = params->value(name, "ny");
   int numFeatures = (int) params->value(name, "nf");

   if (params->present(name, "nBorder")) nBorder = (int) params->value(name, "nBorder");

   float xScalef = log2f(parent->width() / nx);
   float yScalef = log2f(parent->height() / ny);

   int xScale = (int) nearbyintf(xScalef);
   int yScale = (int) nearbyintf(yScalef);

   clayer = pvlayer_new(name, xScale, yScale, (int)nx, (int)ny, numFeatures, nBorder);
   clayer->layerType = type;

   float width  = nBorder;
   float height = (clayer->loc.nx > clayer->loc.ny) ? clayer->loc.nx : clayer->loc.ny;
   int numBorderItems = (int) width * (int) height * clayer->numFeatures;

   // calculate maximum size of a border cube
   maxBorderSize = pvcube_size(numBorderItems);

   return 0;
}

int HyPerLayer::initBorder(PVLayerCube * border, int borderId)
{
   // TODO - this uses clayer nxGlobal and nyGlobal
   // TODO - is this correct, kx0 or ky0 < 0
   // TODO - does global patch need to expand to take into account border regions (probably)

   PVLayerLoc loc = clayer->loc;
   int numBorder = clayer->numBorder;

   switch (borderId) {
   case NORTHWEST:
      loc.nx = numBorder;
      loc.ny = numBorder;
      loc.kx0 = clayer->loc.kx0 - numBorder;
      loc.ky0 = clayer->loc.ky0 - numBorder;
      break;
   case NORTH:
      loc.ny = numBorder;
      loc.ky0 = clayer->loc.ky0 - numBorder;
      break;
   case NORTHEAST:
      loc.nx = numBorder;
      loc.ny = numBorder;
      loc.kx0 = clayer->loc.kx0 + clayer->loc.nx;
      loc.ky0 = clayer->loc.ky0 - numBorder;
      break;
   case WEST:
      loc.nx = numBorder;
      loc.kx0 = clayer->loc.kx0 - numBorder;
      break;
   case EAST:
      loc.nx = numBorder;
      loc.kx0 = clayer->loc.kx0 + clayer->loc.nx;
      break;
   case SOUTHWEST:
      loc.nx = numBorder;
      loc.ny = numBorder;
      loc.kx0 = clayer->loc.kx0 - numBorder;
      loc.ky0 = clayer->loc.ky0 + clayer->loc.ny;
      break;
   case SOUTH:
      loc.ny = numBorder;
      loc.ky0 = clayer->loc.ky0 + clayer->loc.ny;
      break;
   case SOUTHEAST:
      loc.nx = numBorder;
      loc.ny = numBorder;
      loc.kx0 = clayer->loc.kx0 + clayer->loc.nx;
      loc.ky0 = clayer->loc.ky0 + clayer->loc.ny;
      break;
   default:
      fprintf(stderr, "ERROR:HyPerLayer:initBorder: bad border index %d\n", borderId);
   }

   pvcube_init(border, &loc, (int) loc.nx * (int) loc.ny * clayer->numFeatures);

   return 0;
}

int HyPerLayer::initGlobal(int colId, int colRow, int colCol, int nRows, int nCols)
{
   return pvlayer_initGlobal(clayer, colId, colRow, colCol, nRows, nCols);
}

int HyPerLayer::columnWillAddLayer(InterColComm * comm, int layerId)
{
   setLayerId(layerId);

   // complete initialization now that we have a parent and a communicator
   // WARNING - this must be done before addPublisher is called
   int id = parent->columnId();
   initGlobal(id, comm->commRow(id), comm->commColumn(id),
                  comm->numCommRows(), comm->numCommColumns());

   comm->addPublisher(this, clayer->activity->size, maxBorderSize, MAX_F_DELAY);

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
   const int nx = (int) clayer->loc.nx;
   const int ny = (int) clayer->loc.ny;
   const int nf = clayer->numFeatures;
   const int numBorder = clayer->numBorder;

   switch (borderId) {
   case 0:
      numNeurons = clayer->numNeurons;           break;
   case NORTHWEST:
      numNeurons = numBorder * numBorder * nf;   break;
   case NORTH:
      numNeurons = nx * numBorder * nf;          break;
   case NORTHEAST:
      numNeurons = numBorder * numBorder * nf;   break;
   case WEST:
      numNeurons = ny * numBorder * nf;          break;
   case EAST:
      numNeurons = ny * numBorder * nf;          break;
   case SOUTHWEST:
      numNeurons = numBorder * numBorder * nf;   break;
   case SOUTH:
      numNeurons = nx * numBorder * nf;          break;
   case SOUTHEAST:
      numNeurons = numBorder * numBorder * nf;   break;
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

int HyPerLayer::recvSynapticInput(HyPerConn * conn, PVLayerCube * activity, int neighbor)
{
   const int numActive = activity->numItems;

   // TODO - handle neighbors
   if (neighbor != 0) {
      return 0;
   }

#ifdef MULTITHREADED
   pv_signal_threads_recv(activity, (unsigned char) neighbor);
   pv_signal_threads_recv(conn->weights(), 0);
   pv_signal_threads_recv(conn->cliques(), 0);
#endif

   for (int kPre = 0; kPre < numActive; kPre++) {
      float a = activity->data[kPre];
      if (a == 0.0f) continue;  // TODO - assume activity is sparse so make this common branch

      PVSynapseBundle * tasks = conn->tasks(kPre, neighbor);
      for (unsigned int i = 0; i < tasks->numTasks; i++) {
         PVSynapseTask * task = tasks->tasks[i];
         PVPatch * phi = task->data;
         PVPatch * weights = task->weights;

      // WARNING - assumes weight and phi patches from task same size
      //         - assumes patch stride sf is 1

         int nk  = (int) phi->nf * (int) phi->nx;
         int ny  = (int) phi->ny;
         int sy  = (int) phi->sy;
         int syw = (int) weights->sy;

         // TODO - unroll
         for (int y = 0; y < ny; y++) {
            pvpatch_accumulate(nk, phi->data + y*sy, a, weights->data + y*syw);
//          if (err != 0) printf("  ERROR kPre = %d\n", kPre);
         }
      }
   }

   return 0;
}

int HyPerLayer::publish(InterColComm* comm, float time)
{
   comm->publish(this, clayer->activity);
   if (outputOnPublish) outputState(time);
   return 0;
}

int HyPerLayer::insertProbe(PVLayerProbe * p)
{
   PVLayerProbe ** tmp;
   tmp = (PVLayerProbe **) malloc((numProbes + 1) * sizeof(PVLayerProbe *));

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   delete probes;

   probes = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerLayer::outputState(float time)
{
   char str[32];

   const int nx = (int) clayer->loc.nx;
   const int ny = (int) clayer->loc.ny;
   const int nf = clayer->numFeatures;

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(time, clayer);
   }

   // Output spike events and V
   sprintf(str, "f%1.1d", clayer->layerId);
   pv_dump(str, ioAppend, clayer->activity->data, nx, ny, nf);
   pv_dump_sparse(str, ioAppend, clayer->activity->data, nx, ny, nf);
   sprintf(str, "V%1.1d", clayer->layerId);
   pv_dump(str, ioAppend, clayer->V, nx, ny, nf);

   // append to dump file after original open
   this->ioAppend = 1;

   return 0;
}

int HyPerLayer::writeState(const char * path, float time)
{
   char fullpath[PV_PATH_MAX];

   // print activity
   sprintf(fullpath, "%s/f%1.1d.tif", path, clayer->layerId);
   pv_tiff_write_cube(fullpath, clayer->activity, (int)clayer->loc.nx, (int)clayer->loc.ny, clayer->numFeatures);

   return 0;
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

int HyPerLayer::getActivityLength(void)
{
   return getCLayer()->numNeurons;
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
   int nx = (int) dest->loc.nx;
   int ny = (int) dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * (int) src->loc.nx;
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
   int ny = (int) dest->loc.ny;
   int sy = clayer->numFeatures * (int) dest->loc.nx;

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
   int nx = (int) dest->loc.nx;
   int ny = (int) dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * (int) src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src ->data + ((int) src->loc.nx - 1)*nf;
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
   int nx = (int) dest->loc.nx;
   int ny = (int) dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * (int) src->loc.nx;
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
   int nx = (int) dest->loc.nx;
   int ny = (int) dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * (int) src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src ->data + ((int) src->loc.nx - 1)*nf;
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
   int nx = (int) dest->loc.nx;
   int ny = (int) dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * (int) src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src-> data + ((int) src->loc.ny - 1)*sySrc;
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
   int ny = (int) dest->loc.ny;
   int sy = clayer->numFeatures * (int) dest->loc.nx;

   pvdata_t * src0 = src ->data + ((int) src->loc.ny - 1)*sy;
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
   int nx = (int) dest->loc.nx;
   int ny = (int) dest->loc.ny;
   int nf = clayer->numFeatures;

   int sySrc = nf * (int) src->loc.nx;
   int syDst = nf * nx;

   pvdata_t * src0 = src-> data + ((int) src->loc.nx - 1)*nf
                                + ((int) src->loc.ny - 1)*sySrc;
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
   int nx = (int) dest->loc.nx;
   int ny = (int) dest->loc.ny;
   int x  = (int) src->loc.nx - nx;
   int y  = (int) src->loc.ny - ny;
   int off = x + y * (int) src->loc.nx;

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
      float sf = 1;
      float sx = nf;
      float sy = sx * nx;

      PVPatch * p = (PVPatch *) malloc(sizeof(PVPatch));
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
      float sf = 1;
      float sx = nf;
      float sy = sx * nx;

      size_t dataSize = nx * ny * nf * sizeof(float);
      PVPatch * p = (PVPatch *) malloc(sizeof(PVPatch) + dataSize);
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
   const float nx = phi->nx;
   const float ny = phi->ny;
   const float nf = phi->nf;
   const float sy = phi->sy;
   const float sf = phi->sf;

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
#endif

/**
 * Return the _global_ leading index in a direction of a patch in the post layer
 *   NOTE: float OK size for kxPre because only k index in a specific direction
 * @kPre is the _global_ pre-synaptic index in a direction
 * @k0Post is the index offset in the post layer
 * @scale is the difference in size scale (2^scale) between post and pre layers
 * @nPatch is the size of patch in a given direction
 * @nLocal is the local size of layer in a direction
 */
float pvlayer_patchHead(float kxPre, float kxPost0Left, int xScale, float nPatch)
{
   float shift = 0;
   if ((int) nPatch % 2 == 0) {
      // if even, can't shift evenly (at least for scale < 0)
      // the later choice alternates direction so not always to left
      shift = (xScale < 0) ? 1 : (int) kxPre % 2;
   }
   shift -= (int) (0.5 * nPatch);
   return kxPost0Left + shift + nearby_neighbor((int) kxPre, xScale);

#ifdef SHIFTED_CENTERS
   // this works better? for nPatch even, not so well for odd
   if (xScale == 0 && ((int)nPatch % 2) == 1) {
      return kxPost0Left + kxPre + 0.5*(1 - nPatch);
   }
   else {
      float a = powf(2.0f,-1.0f*xScale);
      return floorf((kxPost0Left + a*kxPre) + 0.5*(1 - nPatch));
      //      return floorf((kxPost0Left + a*kxPre) + (1.5f - 0.5f*nPatch));
   }
#endif
}

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

