/*
 * Patterns.cpp
 *
 *  Created on: April 21, 2010
 *      Author: Craig Rasmussen
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

   // set switching and moving probabilities
   pSwitch = 0.0;
   pMove = 0.0;

   // check for explicit parameters in params.stdp
   PVParams * params = hc->parameters();
   if (params->present(name, "pMove")) {
      pMove = params->value(name, "pMove");
      //printf("pMove = %f\n", pMove);
   }

   if (params->present(name, "pSwitch")) {
      pSwitch = params->value(name, "pSwitch");
      //printf("pSwitch = %f\n",pSwitch);
   }

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages",0);

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
   // extended frame
   const PVLayerLoc * loc = getLayerLoc();

   const int nx = loc->nx + 2 * loc->nPad;
   const int ny = loc->ny + 2 * loc->nPad;
   const int sx = 1;
   const int sy = sx * nx;

   const int width = 16;
   const int height = 16;
   const int interval = 4;
   int x, y;

   // reset data buffer
   const int nk = nx * ny;
   for (int k = 0; k < nk; k++) {
      data[k] = 0.0;
   }

   if (type == RECTANGLES) {
      const int y0 = (ny - height) / 2;
      const int x0 = (nx - width)  / 2;
      for (int iy = y0; iy < y0 + height; iy++) {
         for (int ix = x0; ix < x0 + width; ix++) {
            data[ix * sx + iy * sy] = val;
         }
      }
      return 0;
   }

   if (orientation == vertical) { // vertical bars
      for (int iy = 0; iy < ny; iy++) {
         for (int ix = position; ix < nx + position; ix += interval) {
            for (int m = 0; m < width; m++) {
               x = (ix + m) % nx;
               data[x * sx + iy * sy] = val;
            }
         }
      }
   }
   else { // horizontal bars
      for (int ix = 0; ix < nx; ix++) {
         for (int iy = position; iy < ny + position; iy += interval) {
            for (int m = 0; m < width; m++) {
               y = (iy + m) % ny;
               data[ix * sx + y * sy] = val;
            }
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
   // alternate between vertical and horizontal bars
   double p = pv_random_prob();

   if (orientation == vertical) { // current vertical gratings
      if (p < pSwitch) { // switch with probability pSwitch
         orientation = horizontal;
         initPattern(MAXVAL);
      }
   }
   else {
      if (p < pSwitch) { // current horizontal gratings
         orientation = vertical;
         initPattern(MAXVAL);
      }
   }

   // moving probability
   double p_move = pv_random_prob();
   if (p_move < pMove) {
      //position = calcPosition(position, nx);
      //position = (start++) % 4;
      position = (int) (4.0*pv_random_prob());
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
   else if (random_jump){
      for (int i = 0; i < step; i++) {
         if ((i * dp < p) && (p < (i + 1) * dp)) {
            return i;
         }
      }
   }
   return pos;
}

} // namespace PV
