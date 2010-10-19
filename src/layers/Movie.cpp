/*
 * Movie.cpp
 *
 *  Created on: Sep 25, 2009
 *      Author: travel
 */

#include "Movie.hpp"
#include "../io/imageio.hpp"
#include "../utils/pv_random.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

namespace PV {

Movie::Movie(const char * name, HyPerCol * hc, const char * fileOfFileNames, float displayPeriod)
     : Image(name, hc)
{
   PVLayerLoc * loc = &clayer->loc;

   this->displayPeriod = displayPeriod;
   this->nextDisplayTime = hc->simulationTime() + displayPeriod;

   fp = fopen(fileOfFileNames, "r");
   assert(fp != NULL);

   filename = strdup(getNextFileName());
   assert(filename != NULL);

   // get size info from image so that data buffer can be allocated
   int status = getImageInfo(filename, parent->icCommunicator(), &imageLoc);
   assert(status == 0);

   // create mpi_datatypes for border transfer
   mpi_datatypes = Communicator::newDatatypes(loc);

//   int N = imageLoc.nx * imageLoc.ny * imageLoc.nBands;
//   imageData = new float [N];
//   for (int i = 0; i < N; ++i) {
//      imageData[i] = 0;
//   }
   imageData = NULL;

   // need all image bands until converted to gray scale
   loc->nBands = imageLoc.nBands;

#ifdef OBSOLETE
   initialize_data(loc);

//   N = loc.nx * loc.ny * loc.nBands;
//   imageData = new float [N];
//   for (int i = 0; i < N; ++i) {
//      imageData[i] = 0;
//   }
//
#endif

   this->offsetX = imageLoc.nx / 2;
   this->offsetY = imageLoc.ny / 2;

   read(filename, offsetX, offsetY);

   // for now convert images to grayscale
   if (loc->nBands > 1) {
      toGrayScale();
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
   return clayer->loc;
}

int Movie::updateState(float time, float dt)
{
  updateImage(time, dt);
  return 0;
}

/**
 * update the image buffers
 *
 * return true if buffers have changed
 */
bool Movie::updateImage(float time, float dt)
{
   PVLayerLoc * loc = &clayer->loc;

   if (time >= nextDisplayTime) {
      if (filename != NULL) free(filename);
      filename = strdup(getNextFileName());
      assert(filename != NULL);
      nextDisplayTime += displayPeriod;
   }

   // need all image bands until converted to gray scale
   loc->nBands = imageLoc.nBands;

   int step = 2;
   offsetX = calcPosition(offsetX, step, imageLoc.nx - loc->nx);
   offsetY = calcPosition(offsetY, step, imageLoc.ny - loc->nx);

   read(filename, offsetX, offsetY);

   // for now convert images to grayscale
   if (loc->nBands > 1) {
      toGrayScale();
   }

   // exchange border information
   exchange();

   lastUpdateTime = time;

   return true;
}

int Movie::outputState(float time, bool last)
{
   if (writeImages) {
      char basicFilename[PV_PATH_MAX + 1];
      snprintf(basicFilename, PV_PATH_MAX, "./output/Movie_%f.tif", time);
      write(basicFilename);
   }
   return 0;
}

int Movie::copyReducedImagePortion()
{
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx;
   const int ny = loc->ny;

   const int nx0 = imageLoc.nx;
   const int ny0 = imageLoc.ny;

   assert(nx0 <= nx);
   assert(ny0 <= ny);

   const int i0 = nx/2 - nx0/2;
   const int j0 = ny/2 - ny0/2;

   int ii = 0;
   for (int j = j0; j < j0+ny0; j++) {
      for (int i = i0; i < i0+nx0; i++) {
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

/**
 * Return an integer between 0 and (step-1)
 */
int Movie::calcPosition(int pos, int step, int sizeLength)
{
   const float dp = 1.0 / step;
   const double p = pv_random_prob();
   const int random_walk = 0;
   const int move_forward = 0;
   const int move_backward = 0;
   const int random_jump = 1;

   if (random_walk) {
      if (p < 0.5) {
         pos += step;
      } else {
         pos -= step;
      }
   } else if (move_forward) {
      pos += step;
   } else if (move_backward) {
      pos -= step;
   } else if (random_jump) {
      for (int i = 0; i < step; i++) {
         if ((i * dp < p) && (p < (i + 1) * dp)) {
            pos = (pv_random_prob() < 0.5) ? pos - i : pos + i;
         }
      }
   }

   pos = (pos < 0) ? -pos : pos;
   pos = (pos > sizeLength) ? sizeLength - (pos-sizeLength) : pos;

   return pos;
}

}
