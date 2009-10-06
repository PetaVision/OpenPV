/*
 * Image.cpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#include "Image.hpp"
#include "../io/imageio.hpp";

#include <assert.h>
#include <string.h>

namespace PV {

Image::Image(const char * name, HyPerCol * hc)
{
   initialize_base(name, hc);
}

Image::Image(const char * name, HyPerCol * hc, const char * filename)
{
   initialize_base(name, hc);

   // get size info from image so that data buffer can be allocated
   int status = getImageInfo(filename, comm, &loc);

   // create mpi_datatypes for border transfer
   mpi_datatypes = Communicator::newDatatypes(&loc);

   if (status) return;

   const int N = (loc.nx + 2*loc.nPad) * (loc.ny + 2*loc.nPad) * loc.nBands;
   data = new float [N];

   for (int i = 0; i < N; ++i) {
      data[i] = 0;
   }

   read(filename);

   // for now convert images to grayscale
   if (loc.nBands > 1) {
      this->toGrayScale();
   }

   // exchange border information
   exchange();
}

Image::~Image()
{
   free(name);

   if (data != NULL) {
      delete data;
      data = NULL;
   }
}

int Image::initialize_base(const char * name, HyPerCol * hc)
{
   this->name = strdup(name);
   this->data = NULL;
   this->comm = hc->icCommunicator();
   mpi_datatypes = NULL;

   PVParams * params = hc->parameters();

   loc.nx       = 0;   loc.ny       = 0;
   loc.nxGlobal = 0;   loc.nyGlobal = 0;
   loc.kx0      = 0;   loc.ky0      = 0;
   loc.nPad     = 0;   loc.nBands   = 0;

   if (params->present(name, "marginWidth")) {
      loc.nPad = (int) params->value(name, "marginWidth");
   }

   return 0;
}

pvdata_t * Image::getImageBuffer()
{
   return data;
}

LayerLoc Image::getImageLoc()
{
   return loc;
}

/**
 * update the image buffers
 *
 * return true if buffers have changed
 */
bool Image::updateImage(float time, float dt)
{
   // default is to do nothing for now
   // eventually could go through a list of images
   return false;
}

int Image::read(const char * filename)
{
   int status = 0;

   const int n = loc.nx * loc.ny * loc.nBands;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   // read the image and scatter the local portions
   status = scatterImageFile(filename, comm, &loc, buf);

   if (status == 0) {
      status = copyFromInteriorBuffer(buf);
   }
   delete buf;

   return status;
}

int Image::write(const char * filename)
{
   int status = 0;

   const int n = loc.nx * loc.ny * loc.nBands;
   unsigned char * buf = new unsigned char[n];
   assert(buf != NULL);

   status = copyToInteriorBuffer(buf);

   // gather the local portions and write the image
   status = gatherImageFile(filename, comm, &loc, buf);

   delete buf;

   return status;
}

int Image::exchange()
{
   return comm->exchange(data, mpi_datatypes, &loc);
}

int Image::copyToInteriorBuffer(unsigned char * buf)
{
   const int nx = loc.nx;
   const int ny = loc.ny;

   const int nxBorder = loc.nPad;
   const int nyBorder = loc.nPad;

   const int sy = nx + 2*nxBorder;
   const int sb = sy * (ny + 2*nyBorder);

   int ii = 0;
   for (int b = 0; b < loc.nBands; b++) {
      for (int j = 0; j < ny; j++) {
         int jex = j + nyBorder;
         for (int i = 0; i < nx; i++) {
            int iex = i + nxBorder;
            buf[ii++] = (unsigned char) data[iex + jex*sy + b*sb];
         }
      }
   }
   return 0;
}

int Image::copyFromInteriorBuffer(const unsigned char * buf)
{
   const int nx = loc.nx;
   const int ny = loc.ny;

   const int nxBorder = loc.nPad;
   const int nyBorder = loc.nPad;

   const int sy = nx + 2*nxBorder;
   const int sb = sy * (ny + 2*nyBorder);

   int ii = 0;
   for (int b = 0; b < loc.nBands; b++) {
      for (int j = 0; j < ny; j++) {
         int jex = j + nyBorder;
         for (int i = 0; i < nx; i++) {
            int iex = i + nxBorder;
            data[iex + jex*sy + b*sb] = (pvdata_t) buf[ii++];
         }
      }
   }
   return 0;
}

int Image::toGrayScale()
{
   const int nx_ex = loc.nx + 2*loc.nPad;
   const int ny_ex = loc.ny + 2*loc.nPad;

   const int numBands = loc.nBands;

   const int sx = 1;
   const int sy = nx_ex;
   const int sb = nx_ex * ny_ex;

   if (numBands < 2) return 0;

   for (int j = 0; j < ny_ex; j++) {
      for (int i = 0; i < nx_ex; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = data[i*sx + j*sy + b*sb];
            val += d*d;
//            val += d;
         }
         data[i*sx + j*sy + 0*sb] = sqrt(val/numBands);
//         data[i*sx + j*sy + 0*sb] = val/numBands;
      }
   }

   loc.nBands = 1;

   return 0;
}

int Image::convertToGrayScale(LayerLoc * loc, unsigned char * buf)
{
   const int nx = loc->nx;
   const int ny = loc->ny;

   const int numBands = loc->nBands;

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
         buf[i*sx + j*sy + 0*sb] = sqrt(val/numBands);
      }
   }

   loc->nBands = 1;

   return 0;
}

int Image::convolution()
{
   const int nx = loc.nx;
   const int ny = loc.ny;
   const int numBands = loc.nBands;

   const int sx = 1;
   const int sy = nx;
   const int sb = nx * ny;

   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         float val = 0;
         for (int b = 0; b < numBands; b++) {
            float d = data[i*sx + j*sy + b*sb];
//            val += d*d;
            val += d;
         }
//         data[i*sx + j*sy + 0*sb] = sqrt(val)/numBands;
         data[i*sx + j*sy + 0*sb] = val/numBands;
      }
   }

   loc.nBands = 1;

   const int nPad = 15;
   const int nPad_2 = nPad/2;

   float * buf = new float[nx*ny];
   for (int i = 0; i < nx*ny; i++) buf[i] = 0;

   float max = -1.0e9;
   float min = -max;

   for (int j = nPad_2; j < ny-nPad_2; j++) {
      for (int i = nPad_2; i < nx-nPad_2; i++) {
         float av = 0;
         float sq = 0;
         for (int kj = 0; kj < nPad; kj++) {
            for (int ki = 0; ki < nPad; ki++) {
               int ix = i + ki - nPad_2;
               int iy = j + kj - nPad_2;
               float val = data[ix*sx + iy*sy];
               av += val;
               sq += val * val;
            }
         }
         av = av / (nPad*nPad);
         if (av < min) min = av;
         if (av > max) max = av;
         sq  = sqrt( sq/(nPad*nPad) - av*av ) + tau;
//         buf[i*sx + j*sy] = data[i*sx + j*sy] + mid - av;
         buf[i*sx + j*sy] = .95*255 * (data[i*sx + j*sy] - .95*av) / sq;
      }
   }

   printf("min==%f max==%f\n", min, max);

   for (int i = 0; i < nx*ny; i++) data[i] = buf[i];

   return 0;
}

} // namespace PV
