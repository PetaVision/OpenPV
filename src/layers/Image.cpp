/*
 * Image.cpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#include "Image.hpp"
#include "../io/imageio.hpp";

namespace PV {

Image::Image(const char * filename, HyPerCol * hc)
{
   this->data = NULL;
   this->comm = hc->icCommunicator();

   loc.nx       = 0;   loc.ny       = 0;
   loc.nxGlobal = 0;   loc.nyGlobal = 0;
   loc.kx0      = 0;   loc.ky0      = 0;
   loc.nPad     = 0;   loc.nBands   = 0;

   // get size info from image so that data buffer can be allocated
   int status = getImageInfo(filename, comm, &loc);

   if (status) return;

   const int N = loc.nx * loc.ny * loc.nBands;
   data = new float [N];

   for (int i = 0; i < N; ++i) data[i] = 0;

   read(filename);
}

Image::~Image()
{
   if (data != NULL) delete data;
}

int Image::read(const char * filename)
{
   int status = 0;

   // read the image and scatter the local portions
   status = scatterImageFile(filename, comm, &loc, data);

   return status;
}

int Image::write(const char * filename)
{
   int status = 0;

   // gather the local portions and write the image
   status = gatherImageFile(filename, comm, &loc, data);

   return status;
}

int Image::toGrayScale()
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
            val += d*d;
//            val += d;
         }
         data[i*sx + j*sy + 0*sb] = sqrt(val)/numBands;
//         data[i*sx + j*sy + 0*sb] = val/numBands;
      }
   }

   loc.nBands = 1;

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
   const float mid = 255./2.;

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
