/*
 * Patterns.cpp
 *
 *  Created on: April 21, 2010
 *      Author: Marian Anghel and Craig Rasmussen
 */

#include "Patterns.hpp"
#include <src/include/pv_common.h>  // for PI
#include <src/utils/pv_random.h>

#define MAXVAL  1.0f

namespace PV {

// CER-new
FILE * fp;
int start = 0;

Patterns::Patterns(const char * name, HyPerCol * hc, PatternType type) :
   Image(name, hc)
{
   // CER-new
   fp = fopen("bar-pos.txt", "w");

   this->type = type;

   // set default params
   // set reference position of bars
   this->prefPosition = 3;
   this->position = this->prefPosition;
   this->lastPosition = this->prefPosition;

   // set bars orientation to default values
   this->orientation = vertical;
   this->lastOrientation = orientation;

   const PVLayerLoc * loc = getLayerLoc();

   // check for explicit parameters in params.stdp
   //
   PVParams * params = hc->parameters();

   minWidth  = 4.0;
   minHeight = 4.0;

   maxWidth  = params->value(name, "width", loc->nx);
   maxHeight = params->value(name, "height", loc->ny);

   pMove   = params->value(name, "pMove", 0.0);
   pSwitch = params->value(name, "pSwitch", 0.0);

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages", 0.0);

   initPattern(MAXVAL);

   // make sure initialization is finished
   updateState(0.0, 0.0);
}

Patterns::~Patterns()
{
   // CER-new
   fclose(fp);
}

int Patterns::tag()
{
   if (orientation == vertical) return position;
   else                         return 10*position;
}

int Patterns::initPattern(float val)
{
   int width, height;
   const int interval = 4;

   // extended frame
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx + 2 * loc->nb;
   const int ny = loc->ny + 2 * loc->nb;
   const int sx = 1;
   const int sy = sx * nx;

   // reset data buffer
   const int nk = nx * ny;
   for (int k = 0; k < nk; k++) {
      data[k] = 0.0;
   }

   if (type == RECTANGLES) {
      width  = minWidth  + (maxWidth  - minWidth)  * pv_random_prob();
      height = minHeight + (maxHeight - minHeight) * pv_random_prob();

      const int half_w = width/2;
      const int half_h = height/2;

      // random center location
      const int xc = (nx-1) * pv_random_prob();
      const int yc = (ny-1) * pv_random_prob();

      const int x0 = (xc - half_w < 0) ? 0 : xc - half_w;
      const int y0 = (yc - half_h < 0) ? 0 : yc - half_h;

      const int x1 = (xc + half_w > nx) ? nx : xc + half_w;
      const int y1 = (yc + half_h > ny) ? ny : yc + half_h;

      for (int iy = y0; iy < y1; iy++) {
         for (int ix = x0; ix < x1; ix++) {
            data[ix * sx + iy * sy] = val;
         }
      }
      position = x0 + y0*nx;
      return 0;
   }

   // type is bars

   if (orientation == vertical) { // vertical bars
      width = maxWidth;
      for (int iy = 0; iy < ny; iy++) {
         for (int ix = 0; ix < nx; ix++) {
            int m = (ix + position) % (2*width);
            data[ix * sx + iy * sy] = (m < width) ? val : 0;
         }
      }
   }
   else { // horizontal bars
      height = maxHeight;
      for (int iy = 0; iy < ny; iy++) {
         int m = (iy + position) % (2*height);
         for (int ix = 0; ix < nx; ix++) {
            data[ix * sx + iy * sy] = (m < height) ? val : 0;
         }
      }
   }

   return 0;
}

/**
 * update the image buffers
 */
int Patterns::updateState(float time, float dt)
{
   int size = 0;

   // alternate between vertical and horizontal bars
   double p = pv_random_prob();

   if (orientation == vertical) { // current vertical gratings
      size = maxWidth;
      if (p < pSwitch) { // switch with probability pSwitch
         orientation = horizontal;
         initPattern(MAXVAL);
      }
   }
   else {
      size = maxHeight;
      if (p < pSwitch) { // current horizontal gratings
         orientation = vertical;
         initPattern(MAXVAL);
      }
   }

   // moving probability
   double p_move = pv_random_prob();
   if (p_move < pMove) {
      position = calcPosition(position, 2*size);
      //position = (start++) % 4;
      //position = prefPosition;
      initPattern(MAXVAL);
      //fprintf(fp, "%d %d %d\n", 2*(int)time, position, lastPosition);
   }
   else {
      position = lastPosition;
   }

   if (lastPosition != position || lastOrientation != orientation) {
      lastPosition = position;
      lastOrientation = orientation;
      lastUpdateTime = time;
      if (writeImages) {
         char basicfilename[PV_PATH_MAX+1]; // is +1 needed?
         snprintf(basicfilename, PV_PATH_MAX, "Bars_%.2f.tif", time);
         write(basicfilename);
      }
      return 1;
   }
   else {
      return 0;
   }
}

/**
 *
 *  Return an integer between 0 and (step-1)
 */
int Patterns::calcPosition(int pos, int step)
{
   float dp = 1.0 / step;
   double p = pv_random_prob();
   int random_walk = 1;
   int move_forward = 0;
   int move_backward = 0;
   int random_jump = 0;

   if (random_walk) {
      if (p < 0.5){
         pos = (pos+1) % step;
      } else {
         pos = (pos-1+step) % step;
      }
      //printf("pos = %f\n",position);
   } else if (move_forward){
      pos = (pos+1) % step;
   } else if (move_backward){
      pos = (pos-1+step) % step;
   }
   else if (random_jump) {
	   pos = int(p * step) % step;
      }

   return pos;
}

} // namespace PV
