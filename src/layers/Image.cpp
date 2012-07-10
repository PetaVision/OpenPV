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

Image::Image() {
   initialize_base();
}

Image::Image(const char * name, HyPerCol * hc, const char * filename) {
   initialize_base();
   initialize(name, hc, filename);
}

Image::~Image() {
   free(filename);
   filename = NULL;
   Communicator::freeDatatypes(mpi_datatypes); mpi_datatypes = NULL;
}

int Image::initialize_base() {
   mpi_datatypes = NULL;
   data = NULL;
   filename = NULL;
   imageData = NULL;
   return PV_SUCCESS;
}

int Image::checkpointRead(float * timef){

   PVParams * params = parent->parameters();
   this->useParamsImage      = (int) params->value(name,"useParamsImage", 0);
   if (this->useParamsImage) {
      fprintf(stderr,"Initializing image from params file location ! \n");
      * timef = parent->simulationTime(); // fakes the pvp time stamp
   }
   else {
      fprintf(stderr,"Initializing image from checkpoint NOT from params file location! \n");
      HyPerLayer::checkpointRead(timef);
   }


   return PV_SUCCESS;
}


int Image::initialize(const char * name, HyPerCol * hc, const char * filename) {
   HyPerLayer::initialize(name, hc, 0);
   int status = PV_SUCCESS;

   PVParams * params = parent->parameters();
   this->writeImages = params->value(name, "writeImages", 0) != 0;
   readOffsets();

   GDALColorInterp * colorbandtypes = NULL;
   if(filename != NULL ) {
      this->filename = strdup(filename);
      assert( this->filename != NULL );
      status = getImageInfo(filename, parent->icCommunicator(), &imageLoc, &colorbandtypes);
      if( getLayerLoc()->nf != imageLoc.nf && getLayerLoc()->nf != 1) {
         fprintf(stderr, "Image %s: file %s has %d features but the layer has %d features.  Exiting.\n",
               name, filename, imageLoc.nf, getLayerLoc()->nf);
         exit(PV_FAILURE);
      }
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
      readImage(filename, offsetX, offsetY, colorbandtypes);
   }
   free(colorbandtypes); colorbandtypes = NULL;

   // exchange border information
   exchange();

   return status;
}

int Image::readOffsets() {
   PVParams * params = parent->parameters();
   this->offsetX      = (int) params->value(name,"offsetX", 0);
   this->offsetY      = (int) params->value(name,"offsetY", 0);
   return PV_SUCCESS;
}

int Image::initializeState() {
   assert(parent->parameters()->value(name, "restart", 0.0f, false)==0.0f); // initializeState should only be called if restart is false
   // Image doesn't use the V buffer so free it and set the pointer to null.
   free(clayer->V);
   clayer->V = NULL;
   return PV_SUCCESS;
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

/**
 * return some useful information about the image
 */
int Image::tag()
{
   return 0;
}

int Image::recvSynapticInput(HyPerConn * conn, const PVLayerCube * cube, int neighbor)
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

int Image::readImage(const char * filename)
{
   return readImage(filename, 0, 0, NULL);
}

int Image::readImage(const char * filename, int offsetX, int offsetY, GDALColorInterp * colorbandtypes)
{
   int status = 0;
   PVLayerLoc * loc = & clayer->loc;

   const int n = loc->nx * loc->ny * imageLoc.nf;
   // Use number of bands in file instead of in params, to allow for grayscale conversion
   float * buf = new float[n];
   assert(buf != NULL);

   // read the image and scatter the local portions
   status = scatterImageFile(filename, offsetX, offsetY, parent->icCommunicator(), loc, buf);
   if( loc->nf == 1 && imageLoc.nf > 1 ) {
      float * graybuf = convertToGrayScale(buf,loc->nx,loc->ny,imageLoc.nf, colorbandtypes);
      delete buf;
      buf = graybuf;
   }
   // now buf is loc->nf by loc->nx by loc->ny

// Scaling by 1/255 moved to scatterImageFileGDAL and the compressed part of scatterImageFilePVP,
// since it is unnecessary for uncompressed PVP files.
//   if (status == 0) {
//      float fac = 1.0f / 255.0f;  // normalize to 1.0
//      status = copyFromInteriorBuffer(buf, fac);
//   }

   if( status == 0 ) copyFromInteriorBuffer(buf, 1.0f);

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


int Image::copyToInteriorBuffer(unsigned char * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const int nBorder = loc->nb;

   for(int n=0; n<getNumNeurons(); n++) {
      int n_ex = kIndexExtended(n, nx, ny, nf, nBorder);
      buf[n] = (unsigned char) (fac * data[n_ex]);
   }
   return 0;
}

int Image::copyFromInteriorBuffer(float * buf, float fac)
{
   const PVLayerLoc * loc = getLayerLoc();
   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;

   const int nBorder = loc->nb;

   for(int n=0; n<getNumNeurons(); n++) {
      int n_ex = kIndexExtended(n, nx, ny, nf, nBorder);
      data[n_ex] = fac*buf[n];
   }
   return 0;
}

float * Image::convertToGrayScale(float * buf, int nx, int ny, int numBands, GDALColorInterp * colorbandtypes)
{
   // even though the numBands argument goes last, the routine assumes that
   // the organization of buf is, bands vary fastest, then x, then y.
   if (numBands < 2) return buf;


   const int sxcolor = numBands;
   const int sycolor = numBands*nx;
   const int sb = 1;

   const int sxgray = 1;
   const int sygray = nx;

   float * graybuf = new float[nx*ny];

   float * bandweight = (float *) malloc(numBands*sizeof(float));
   calcBandWeights(numBands, bandweight, colorbandtypes);

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = buf[i*sxcolor + j*sycolor + b*sb];
            val += d*bandweight[b];
         }
         graybuf[i*sxgray + j*sygray] = val;
      }
   }
   free(bandweight);
   return graybuf;
}

int Image::calcBandWeights(int numBands, float * bandweight, GDALColorInterp * colorbandtypes) {
   int colortype = 0; // 1=grayscale(with or without alpha), return value 2=RGB(with or without alpha), 0=unrecognized
   const GDALColorInterp grayalpha[2] = {GCI_GrayIndex, GCI_AlphaBand};
   const GDALColorInterp rgba[4] = {GCI_RedBand, GCI_GreenBand, GCI_BlueBand, GCI_AlphaBand};
   const float grayalphaweights[2] = {1.0, 0.0};
   const float rgbaweights[4] = {0.30, 0.59, 0.11, 0.0}; // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
   switch( numBands ) {
   case 1:
      bandweight[0] = 1.0;
      colortype = 1;
      break;
   case 2:
      if ( !memcmp(colorbandtypes, grayalpha, 2*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, grayalphaweights, 2*sizeof(float));
         colortype = 1;
      }
      break;
   case 3:
      if ( !memcmp(colorbandtypes, rgba, 3*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, rgbaweights, 3*sizeof(float));
         colortype = 2;
      }
      break;
   case 4:
      if ( !memcmp(colorbandtypes, rgba, 4*sizeof(GDALColorInterp)) ) {
         memcpy(bandweight, rgbaweights, 4*sizeof(float));
         colortype = 2;
      }
      break;
   default:
      break;
   }
   if (colortype==0) {
      equalBandWeights(numBands, bandweight);
   }
   return colortype;
}

void Image::equalBandWeights(int numBands, float * bandweight) {
   float w = 1.0/(float) numBands;
   for( int b=0; b<numBands; b++ ) bandweight[b] = w;
}

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
