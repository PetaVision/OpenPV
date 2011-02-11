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

StringImage::StringImage(const char * name, HyPerCol * hc) : Retina(name, hc)
{
   this->type = type;

   const PVLayerLoc * loc = getLayerLoc();

   // set default params
   // set reference position of bars
   this->position = (loc->nx + 2*loc->nb) / 2.0;
   this->jitter   = 0;

   // set string orientation to default values
   this->orientation = left;
   this->lastOrientation = orientation;

   // check for explicit parameters in params.stdp
   //
   PVParams * params = hc->parameters();

   pMove = params->value(name, "pMove", 0.0);
   pJit  = params->value(name, "pJitter", 0.0);

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages", 0.0);

   initPattern();

   // make sure initialization is finished
   updateState(0.0, 0.0);
}

StringImage::~StringImage()
{
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
   pvdata_t * data = getChannel(CHANNEL_EXC);

   for (int kex = 0; kex < getNumExtended(); kex++) {
      data[kex] = 0.0;
   }
   return 0;
}

/**
 * Set Phi[CHANNEL_EXC] based on string input and then call parent recvSynapticInput
 */
int StringImage::recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor)
{
   // for now alphabet is {i1,i2,f1,f2} (two phases)

   pvdata_t * data = getChannel(CHANNEL_EXC);

   const PVLayerLoc * loc = getLayerLoc();

   int x  = this->position;
   int y  = (loc->ny + 2*loc->nb) / 2;
   int sy = strideYExtended(loc);

   if (pv_random_prob() < pMove) {  // move pattern with probability pMove
      orientation = (orientation == right) ? left : right;
   }
   else if (pv_random_prob() < pJit) {
      jitter = (pv_random_prob() < 0.5) ? 0 : 1;  // pick phase with equal probability
   }

   if (orientation == right) x += 2;
   x += jitter;

   // need to make a tape/string of characters
   // data[x + y*sy] = 1;

   return 0;
}


/**
 * Call recvSynapticInput to set Phi[CHANNEL_EXC] and then let Retina class
 */
int StringImage::updateState(float time, float dt)
{
   int status = recvSynapticInput(NULL, NULL, 0);
   status |= Retina::updateState(time, dt);
   return status;
}

} // namespace PV
