/*
 * Movie.cpp
 *
 *  Created on: Sep 25, 2009
 *      Author: travel
 */

#include "Movie.hpp"
#include "../io/imageio.hpp"

#include <assert.h>
#include <stdio.h>
#include <string.h>

namespace PV {

Movie::Movie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float displayPeriod)
     : Image(name, hc)
{
   this->displayPeriod = displayPeriod;
   this->nextDisplayTime = hc->simulationTime() + displayPeriod;

   fp = fopen(fileOfFileNames, "r");
   assert(fp != NULL);

   const char * filename = getNextFileName();
   assert(filename != NULL);

   // get size info from image so that data buffer can be allocated
   int status = getImageInfo(filename, comm, &imageLoc);
   assert(status == 0);

   // create mpi_datatypes for border transfer
   mpi_datatypes = Communicator::newDatatypes(&loc);

//   int N = imageLoc.nx * imageLoc.ny * imageLoc.nBands;
//   imageData = new float [N];
//   for (int i = 0; i < N; ++i) {
//      imageData[i] = 0;
//   }
   imageData = NULL;

   // need all image bands until converted to gray scale
   loc.nBands = imageLoc.nBands;

   initialize_data(&loc);

//   N = loc.nx * loc.ny * loc.nBands;
//   imageData = new float [N];
//   for (int i = 0; i < N; ++i) {
//      imageData[i] = 0;
//   }
//

   read(filename);
// copyReducedImagePortion();

   // for now convert images to grayscale
   if (loc.nBands > 1) {
      this->toGrayScale();
   }

   // exchange border information
   exchange();
}

Movie::~Movie()
{
   if (imageData != NULL) {
      delete imageData;
      imageData = NULL;
   }
}

pvdata_t * Movie::getImageBuffer()
{
//   return imageData;
   return data;
}

PVLayerLoc Movie::getImageLoc()
{
//   return imageLoc;
   return loc;
}

/**
 * update the image buffers
 *
 * return true if buffers have changed
 */
bool Movie::updateImage(float time, float dt)
{
   if (time < nextDisplayTime) {
      return false;
   }

   nextDisplayTime += displayPeriod;

   const char * filename = getNextFileName();
   assert(filename != NULL);

   // need all image bands until converted to gray scale
   loc.nBands = imageLoc.nBands;

   read(filename);
// copyReducedImagePortion();

   // for now convert images to grayscale
   if (loc.nBands > 1) {
      this->toGrayScale();
   }

   // exchange border information
   exchange();

   lastUpdateTime = time;

   return true;
}

int Movie::copyReducedImagePortion()
{
   int i0, j0, i, j, ii;

   const int nx = loc.nx;
   const int ny = loc.ny;

   const int nx0 = imageLoc.nx;
   const int ny0 = imageLoc.ny;

   assert(nx0 <= nx);
   assert(ny0 <= ny);

   i0 = nx/2 - nx0/2;
   j0 = ny/2 - ny0/2;

   ii = 0;
   for (j = j0; j < j0+ny0; j++) {
      for (i = i0; i < i0+nx0; i++) {
         imageData[ii++] = data[i+nx*j];
      }
   }

   return 0;
}

const char * Movie::getNextFileName()
{
   int c;
   size_t len = PV_PATH_MAX;
   char * path;

   // if at end of file (EOF), rewind

   if ((c = fgetc(fp)) == EOF) {
      rewind(fp);
   }
   else {
      ungetc(c, fp);
   }

   path = fgets(this->inputfile, len, fp);

   if (path != NULL) {
      path[PV_PATH_MAX-1] = '\0';
      len = strlen(path);
      if (len > 1) {
         if (path[len-1] == '\n') {
             path[len-1] = '\0';
         }
      }
   }

   return path;
}

}
