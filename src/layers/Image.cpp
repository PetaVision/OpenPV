/*
 * Image.cpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#include "Image.hpp"
#include "../io/imageio.hpp"

#include <assert.h>
#include <string.h>

namespace PV {

Image::Image(const char * name, HyPerCol * hc)
     : HyPerLayer(name, hc, 0)
{
   initialize(TypeImage);
   initializeImage(NULL);
}

Image::Image(const char * name, HyPerCol * hc, const char * filename)
     : HyPerLayer(name, hc, 0)
{
   initialize(TypeImage);
   initializeImage(filename);
}

Image::~Image()
{
   if (filename != NULL) free(filename);
}

/**
 * data lives in an extended frame of size
 * (nx+2*nPad)*(ny+2*nPad)*nBands
 */
int Image::initializeImage(const char * filename)
{
   int status = 0;

   this->writeImages = (int) parent->parameters()->value(name, "writeImages", 0);
   this->useGrayScale = parent->parameters()->value(name,"useGrayScale",1) != 0;

   if (filename != NULL) {
      this->filename = strdup(filename);
      status = getImageInfo(filename, parent->icCommunicator(), &imageLoc);
   }
   else {
      this->filename = NULL;
      this->imageLoc = * getLayerLoc();
   }
   this->lastUpdateTime = 0.0;

// TODO - must make image conform to layer size

   data = clayer->activity->data;

   // create mpi_datatypes for border transfer
   mpi_datatypes = Communicator::newDatatypes(getLayerLoc());

   if (filename != NULL) {
      read(filename);
   }

   // convert images to grayscale if useGrayScale is set
   if (useGrayScale) {
      this->toGrayScale();
   }

   // exchange border information
   exchange();

   return status;
}

#ifdef PV_USE_OPENCL
// no need for threads for now for image
//
int Image::initializeThreadBuffers()
{
   return CL_SUCCESS;
}

// no need for threads for now for image
//
int Image::initializeThreadKernels()
{
   return CL_SUCCESS;
}
#endif

pvdata_t * Image::getImageBuffer()
{
   return data;
}

PVLayerLoc Image::getImageLoc()
{
   return imageLoc;
}

/**
 * return some useful information about the image
 */
int Image::tag()
{
   return 0;
}

int Image::recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor)
{
   // this should never be called as an image shouldn't have an incoming connection
   recvsyn_timer->start();
   recvsyn_timer->stop();
   return 0;
}

/**
 * update the image buffers
 */
int Image::updateState(float time, float dt)
{
   // make sure image is copied to activity buffer
   //
   update_timer->start();
   update_timer->stop();
   return 0;
}

int Image::outputState(float time, bool last)
{
   // this could probably use Marion's update time interval
   // for some classes
   //
   return 0;
}

//! CLEAR IMAGE
/*!
 * this is Image specific.
 */
int Image::clearImage()
{
   // default is to do nothing for now
   // it could, for example, set the data buffer to zero.

   return 0;
}

int Image::read(const char * filename)
{
   return read(filename, 0, 0);
}

int Image::read(const char * filename, int offsetX, int offsetY)
{
   int status = 0;
   PVLayerLoc * loc = & clayer->loc;

   const int n = loc->nx * loc->ny * loc->nf;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   // read the image and scatter the local portions
   status = scatterImageFile(filename, offsetX, offsetY, parent->icCommunicator(), loc, buf);

   if (status == 0) {
      float fac = 1.0f / 255.0f;  // normalize to 1.0
      status = copyFromInteriorBuffer(buf, fac);
   }
   delete buf;

   return status;
}

/**
 *
 * The data buffer lives in the extended space. Here, we only copy the restricted space
 * to the buffer buf. The size of this buffer is the size of the image patch - borders
 * are not included.
 *
 */
int Image::write(const char * filename)
{
   int status = 0;
   const PVLayerLoc * loc = getLayerLoc();

   const int n = loc->nx * loc->ny * loc->nf;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   status = copyToInteriorBuffer(buf, 255.0);

   // gather the local portions and write the image
   status = gatherImageFile(filename, parent->icCommunicator(), loc, buf);

   delete buf;

   return status;
}

int Image::exchange()
{
   return parent->icCommunicator()->exchange(data, mpi_datatypes, getLayerLoc());
}

#ifdef OBSOLETE
int Image::gatherToInteriorBuffer(unsigned char * buf)
{
   return HyPerLayer::gatherToInteriorBuffer(buf);
   const PVLayerLoc * loc = getLayerLoc();

   assert(loc->nf == 1);

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int nxBorder = loc->nb;
   const int nyBorder = loc->nb;

   const size_t sy = strideY(loc);
   const int sb = sy * (ny + loc->halo.dn + loc->halo.up);

   // only interior portion of local data needed
   //
   unsigned char * srcBuf = (unsigned char *) malloc(nx * ny * sizeof(unsigned char));
   assert(srcBuf != NULL);

   int ii = 0;
   for (int b = 0; b < loc->nf; b++) {
      for (int j = 0; j < ny; j++) {
         int jex = j + nyBorder;
         for (int i = 0; i < nx; i++) {
            int iex = i + nxBorder;
            srcBuf[ii++] = (unsigned char) (255.0f * data[iex + jex*sy + b*sb]);
         }
      }
   }

   gather(parent->icCommunicator(), loc, buf, srcBuf);

   free(srcBuf);

   return 0;
}
#endif

int Image::copyToInteriorBuffer(unsigned char * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   const int nxBorder = loc->nb;
   const int nyBorder = loc->nb;

   const size_t sy = nx + loc->halo.lt + loc->halo.rt;
   const size_t sb = sy * (ny + loc->halo.dn + loc->halo.up);

   int ii = 0;
   for (int b = 0; b < loc->nf; b++) {
      for (int j = 0; j < ny; j++) {
         int jex = j + nyBorder;
         for (int i = 0; i < nx; i++) {
            int iex = i + nxBorder;
            buf[ii++] = (unsigned char) (fac * data[iex + jex*sy + b*sb]);
         }
      }
   }
   return 0;
}

int Image::copyFromInteriorBuffer(const unsigned char * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;

   const int nxBorder = loc->nb;
   const int nyBorder = loc->nb;

   const size_t sy = nx + loc->halo.lt + loc->halo.rt;
   const size_t sb = sy * (ny + loc->halo.dn + loc->halo.up);

   int ii = 0;
   for (int b = 0; b < loc->nf; b++) {
      for (int j = 0; j < ny; j++) {
         int jex = j + nyBorder;
         for (int i = 0; i < nx; i++) {
            int iex = i + nxBorder;
            data[iex + jex*sy + b*sb] = fac * (pvdata_t) buf[ii++];
         }
      }
   }
   return 0;
}

int Image::toGrayScale()
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx_ex = loc->nx + 2*loc->nb;
   const int ny_ex = loc->ny + 2*loc->nb;

   const int numBands = loc->nf;

   const size_t sx = 1;
   const size_t sy = loc->nx + loc->halo.lt + loc->halo.rt;
   const size_t sb = sy * (loc->ny + loc->halo.dn + loc->halo.up);

   if (numBands < 2) return 0;

   for (int j = 0; j < ny_ex; j++) {
      for (int i = 0; i < nx_ex; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = data[i*sx + j*sy + b*sb];
            val += d*d;
//            val += d;
         }
         // store the converted image in the first color band
         data[i*sx + j*sy + 0*sb] = sqrtf(val/numBands);
//         data[i*sx + j*sy + 0*sb] = val/numBands;
      }
   }

   // turn off the color
   clayer->loc.nf = 1;
   clayer->numNeurons = clayer->loc.nx * clayer->loc.ny;
   clayer->numExtended = (clayer->loc.nx + 2*clayer->loc.nb) * (clayer->loc.ny + 2*clayer->loc.nb);

   return 0;
}

// convertToGrayScale() seems broken since it doesn't use loc->nb,
// and no other routine in trunk calls it.
#ifdef OBSOLETE // Marked obsolete July 6, 2011
int Image::convertToGrayScale(PVLayerLoc * loc, unsigned char * buf)
{
   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nf;

   const int sx = 1;
   const int sy = nx;
   const int sb = nx * ny;

   if (numBands < 2) return 0;

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = buf[i*sx + j*sy + b*sb];
            val += d*d;
         }
         // store the converted image in the first color band
         buf[i*sx + j*sy + 0*sb] = (unsigned char) sqrtf(val/numBands);
      }
   }

   // turn off the color
   loc->nf = 1;

   return 0;
}
#endif // OBSOLETE

int Image::convolve(int width)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx_ex = loc->nx + 2*loc->nb;
   const int ny_ex = loc->ny + 2*loc->nb;
   //const int nb = loc->nf;

   const int size_ex = nx_ex * ny_ex;

   // an image is different from normal layers as features (bands) vary last
   const size_t sx = 1;
   const size_t sy = loc->nx + loc->halo.lt + loc->halo.rt;

   const int npx = width;
   const int npy = width;
   const int npx_2 = width/2;
   const int npy_2 = width/2;

   assert(npx <= loc->nb);
   assert(npy <= loc->nb);

   float * buf = new float[size_ex];
   //for (int i = 0; i < size_ex; i++) buf[i] = 0;

   float max = -1.0e9;
   float min = -max;

   // ignore image bands for now
   for (int jex = npy_2; jex < ny_ex - npy_2; jex++) {
      for (int iex = npx_2; iex < nx_ex - npx_2; iex++) {
         float av = 0;
         float sq = 0;
         for (int jp = 0; jp < npy; jp++) {
            for (int ip = 0; ip < npx; ip++) {
   //            int ix = i + ip - npx_2;
   //            int iy = j + jp - npy_2;
   //            float val = data[ix*sx + iy*sy];
   //            av += val;
   //            sq += val * val;
            }
         }
         av = av / (npx*npy);
         min = (av < min) ? av : min;
         max = (av > max) ? av : max;
//         sq  = sqrt( sq/(nPad*nPad) - av*av ) + tau;
//         buf[i*sx + j*sy] = data[i*sx + j*sy] + mid - av;
         buf[iex*sx + jex*sy] = .95f * 255.0f * (data[iex*sx + jex*sy] - .95f * av) / sq;
      }
   }

   printf("min==%f max==%f\n", min, max);

   for (int k = 0; k < size_ex; k++) data[k] = buf[k];

   return 0;
}

} // namespace PV
