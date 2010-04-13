/*
 * Gratings.cpp
 *
 *  Created on: Oct 23, 2009
 *      Author: Marian Anghel
 */

#include "Bars.hpp"
#include "../include/pv_common.h"  // for PI
#include "../utils/pv_random.h"
namespace PV {

Bars::Bars(const char * name, HyPerCol * hc) :
   Image(name, hc)
{
   initialize_data(&loc);

   // set default params
   // set reference position of bars
   this->position = 0;
   this->lastPosition = position;

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

   // set parameters that controls position change
   // default values

   random_walk = 1;
   move_forward = 0;
   move_backward = 0;
   random_jump = 0;

   if (params->present(name, "randomWalk")) {
      int random_walk = params->value(name, "randomWalk");
      //printf("randomWalk = %d\n", random_walk);
   }

   if (params->present(name, "moveForward")) {
      int move_forward = params->value(name, "moveForward");
      //printf("moveForward = %d\n", move_forward);
   }

   if (params->present(name, "moveBackward")) {
      int move_backward = params->value(name, "moveBackward");
      //printf("moveBackward = %d\n", move_backward);
   }

   if (params->present(name, "randomJump")) {
      int random_jump = params->value(name, "randomJump");
      //printf("randomJump = %d\n", random_jump);
   }


   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages",0);

   updateImage(0.0, 0.0);

}

Bars::~Bars()
{
}

/**
 * NOTES:
 *    - Retina calls updateImage(float time, float dt) and expects a bool variable
 *    in return.
 *    - If true, the image has been changed; if false the image has not been
 * changed.
 *    - If true, the retina also calls copyFromImageBuffer() to copy the Image
 *    data buffer into the V buffer (it also normalizes the V buffer so that V <= 1).
 *    - data values here gets scaled  and modulate the spiking probability of
 *    the neurons in the retina. If data has negative values, we can inadvertently
 *    prevent retina neurons from firing. The data should only take positive values,
 *    unless the background retina spiking probability (which gets modulated by scaled
 *    image data is very high. But this is not right. The neurons should have a minimum
 *    spiking rate - which is given by the background rate - and shouldn't have smaller
 *    rate only larger.
 */
bool Bars::updateImage(float time, float dt)
{
   // extended frame
   const int nx = loc.nx + 2 * loc.nPad;
   const int ny = loc.ny + 2 * loc.nPad;
   const int sx = 1;
   const int sy = sx * nx;

   const int width = 1;
   const int step = 6;
   int x, y;

   // alternate between vertical and horizontal bars
   double p = pv_random_prob();

   if (orientation == vertical) { // current vertical gratings
      if (p < pSwitch) { // switch with probability pSwitch
         orientation = horizontal;
      }
   }
   else {
      if (p < pSwitch) { // current horizontal gratings
         orientation = vertical;
      }
   }

   if (orientation == vertical) { // vertical bars

      // moving probability
      double p_move = pv_random_prob();
      if (p_move < pMove) {
         calcPosition(1.0 * nx); // or step
      }
      else {
         position = lastPosition;
      }

      for (int iy = 0; iy < ny; iy++) {
         for (int ix = position; ix < nx + position; ix += width + step) {
            x = (int) fmod(ix, nx);
            for (int m = 0; m < width; m++) {
               data[(x + m) * sx + iy * sy] = 1.0;
            }
            for (int m = 0; m < step; m++) {
               data[(x + width + m) * sx + iy * sy] = 0.0;
            }
         }
      }
   }
   else { // horizontal bars

      // moving probability
      double p_move = pv_random_prob();
      if (p_move < pMove) {
         calcPosition(1.0 * ny);
      }
      else {
         position = lastPosition;
      }

      for (int ix = 0; ix < nx; ix++) {
         for (int iy = position; iy < ny + position; iy += width + step) {
            y = (int) fmod(iy, ny);
            for (int m = 0; m < width; m++) {
               data[ix * sx + (y + m) * sy] = 1.0;
            }
            for (int m = 0; m < step; m++) {
               data[ix * sx + (y + width + m) * sy] = 0.0;
            }
         }
      }
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
      return true;
   }
   else {
      return false;
   }
}

/**
 *
 *  Return an integer between 0 ans (step-1)
 */
void Bars::calcPosition(float step)
{
   float dp = 1.0 / step;
   double p = pv_random_prob();

   if (random_walk) {
      if(p < 0.5){
         position = (int) fmod(position+1, step);
      }else{
         position = (int) fmod(position-1, step);
      }
      //printf("pos = %f\n",position);
   } else if (move_forward){
      position = (int) fmod(position+1, step);
   } else if (move_backward){
      position = (int) fmod(position-1, step);
   }
   else if (random_jump){
      for (int i = 0; i < step; i++) {
         if ((i * dp < p) && (p < (i + 1) * dp)) {
            position = i;
            return;
         }
      }
   }
   return;

}

} // namespace PV
