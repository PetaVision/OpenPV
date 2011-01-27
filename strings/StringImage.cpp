/*
 * StringImage.cpp
 *
 *  Created on: January 27, 2011
 *      Author: Craig Rasmussen
 */

#include "StringImage.hpp"
#include <src/include/pv_common.h>  // for PI
#include <src/utils/pv_random.h>

namespace PV {

StringImage::StringImage(const char * name, HyPerCol * hc) :
   Image(name, hc)
{
   this->type = type;

   const PVLayerLoc * loc = getLayerLoc();

   // set default params
   // set reference position of bars
   this->prefPosition = (loc->nx + 2*loc->nb) / 2.0;
   this->position = this->prefPosition;
   this->lastPosition = this->prefPosition;

   // set bars orientation to default values
   this->orientation = left;
   this->lastOrientation = orientation;

   // check for explicit parameters in params.stdp
   //
   PVParams * params = hc->parameters();

   pMove = params->value(name, "pMove", 0.0);
   pJit  = params->value(name, "pJitter", 0.0);

   pBackground = (.5/1000.) * 50;   // dt==.5 ms, freq=50 Hz

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages", 0.0);

   initPattern();

   // make sure initialization is finished
   updateState(0.0, 0.0);
}

StringImage::~StringImage()
{
// CER-new
//fclose(fp);
}

int StringImage::tag()
{
   if (orientation == left) return    position;
   else                     return 10*position;
}

/**
 * Initialize pattern with background noise
 */
int StringImage::initPattern()
{
   for (int kex = 0; kex < getNumExtended(); kex++) {
      data[kex] = (pv_random_prob() < pBackground) ? 1.0 : 0.0;
   }
   return 0;
}

/**
 * update the image buffers
 */
int StringImage::updateState(float time, float dt)
{
   // for now alphabet is {i1,i2,f1,f2} (two phases)

   // initialize pattern to background
   //
   initPattern();

   const PVLayerLoc * loc = getLayerLoc();

   int x  = this->position;
   int y  = (loc->ny + 2*loc->nb) / 2;
   int sy = strideYExtended(loc);

   if (pv_random_prob() < pMove) {  // move pattern with probability pMove
      orientation = (orientation == right) ? left : right;
      if (orientation == right)   x += 2;
      if (pv_random_prob() < 0.5) x += 1;  // pick phase with even probability
      data[x + y*sy] = 1;
   }
   else if (pv_random_prob() < pJit) {
      if (orientation == right)   x += 2;
	  if (pv_random_prob() < 0.5) x += 1;  // pick phase with even probability
	  data[x + y*sy] = 1;
   }

   return 0;
}

} // namespace PV
