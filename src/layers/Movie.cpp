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

Movie::Movie(const char * fileOfFileNames, HyPerCol * hc, float displayPeriod)
     : Image(hc)
{
   this->displayPeriod = displayPeriod;
   this->nextDisplayTime = hc->simulationTime() + displayPeriod;

   fp = fopen(fileOfFileNames, "r");
   assert(fp != NULL);

   const char * filename = getNextFileName();
   assert(filename != NULL);

   // get size info from image so that data buffer can be allocated
   int status = getImageInfo(filename, comm, &loc);
   assert(status == 0);

   int N = loc.nx * loc.ny * loc.nBands;
   data = new float [N];
   for (int i = 0; i < N; ++i) {
      data[i] = 0;
   }

   // no single input file so must provide (nx,ny) to hc
   imageLoc = hc->getImageLoc();

   N = loc.nx * loc.ny * loc.nBands;
   imageData = new float [N];
   for (int i = 0; i < N; ++i) {
      imageData[i] = 0;
   }

   getReducedImage(filename);
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
   return imageData;
}

LayerLoc Movie::getImageLoc()
{
   return imageLoc;
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

   getReducedImage(filename);

   return true;
}

int Movie::getReducedImage(const char * filename)
{
   int i0, j0, i, j, ii;

   const int nx = loc.nx;
   const int ny = loc.ny;

   const int nx0 = imageLoc.nx;
   const int ny0 = imageLoc.ny;

   assert(nx0 <= nx);
   assert(ny0 <= ny);

   read(filename);

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
