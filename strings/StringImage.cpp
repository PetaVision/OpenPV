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
   const PVLayerLoc * loc = getLayerLoc();

   this->phase  = 0;
   this->jitter = 0;
   this->patternWidth  = 5;   // width of substring before pattern repeats

   // Margins for the string need to be 1/2 the margins for this
   // layer assuming an nxp downward of 3 and because of larger scale
   // for this image. Example, nxScale==4 requires a margin of 4
   // for the simple cell layer. Every simple cell (in the margin)
   // requires a cell on either side, thus margin of 2 for the string.
   strWidth = 1 + parent->localWidth() + loc->nb;  // 1 extra for jitter
   string = new int[strWidth];

   // check for explicit parameters in params.stdp
   //
   PVParams * params = hc->parameters();

   pMove = params->value(name, "pMove", 0.0);
   pJit  = params->value(name, "pJitter", 0.0);

   // set parameters that controls writing of new images
   writeImages = params->value(name, "writeImages", 0.0);

   initializeString();
}

StringImage::~StringImage()
{
   delete [] string;
}

int StringImage::tag()
{
   return phase;
}

/**
 * Initialize string and image data
 */
int StringImage::initializeString()
{
   int kex;
   pvdata_t * data = getChannel(CHANNEL_EXC);

   // string has different size than the layer because the
   // string is effectively the image and this layer is the retina
   for (kex = 0; kex < strWidth; kex++) {
      string[kex] = 0;
   }

   for (kex = 0; kex < strWidth; kex += patternWidth) {
      string[kex] = (pv_random_prob() < 0.5) ? 'a' : 'b';
   }
   phase = 1;  // string[0] position just given a character so advance phase to 1

   for (int kex = 0; kex < getNumExtended(); kex++) {
      data[kex] = 0.0;
   }

   return 0;
}

/**
 * Shift string and update image data
 */
int StringImage::shiftString()
{
   // shift string
   //
   for (int kex = strWidth-1; kex > 0; kex--) {
      string[kex] = string[kex-1];
   }

   // bring in character from left
   //
   if (phase == 0) {
      // character switches with equal probability
      string[0] = (pv_random_prob() < 0.5) ? 'a' : 'b';
   }
   else {
      string[0] = 0;
   }
   phase = (phase+1) % patternWidth;

   return 0;
}

/**
 * Update characters/features in layer data based on string input
 * Assume nxp==3 looking at string and this layer has 8 characters.
 * Working from left fill out data[0:7] = {a(-1), b(-1), a(0), b(0),
 * a(+1), b(+1), blank, blank} based on whether there is an 'a' or 'b'
 * in relative character position in string.
 */
int StringImage::updateLayerData()
{
   int scale = 1.0 / powf(2, getXScale());
   pvdata_t * data = getChannel(CHANNEL_EXC);

   for (int kex = 0; kex < getNumExtended(); kex += scale) {
      int left = jitter + kex/scale;
      int pPos = 0;  // position in pattern
      for (int strPos = left; strPos < left+3; strPos++) {
         if (string[strPos] != 0) {
             data[kex+pPos++] = (string[strPos] == 'a') ? 1 : 0;
             data[kex+pPos++] = (string[strPos] == 'b') ? 1 : 0;
         }
         else {
            data[kex+pPos++] = 0;
            data[kex+pPos++] = 0;

         }
      }
      data[pPos++] = 0;    // blank positions
      data[pPos++] = 0;
   }

   return 0;
}

/**
 * Set Phi[CHANNEL_EXC] based on string input.  This replicates normal response to
 * pre-synaptic activity.
 */
int StringImage::recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor)
{
   if (pv_random_prob() < pMove) {
      shiftString();  // advance tape
   }
   if (pv_random_prob() < pJit) {
      jitter = (1+jitter) % 2;
   }
   return updateLayerData();
}


/**
 * Call recvSynapticInput to set Phi[CHANNEL_EXC] and then let Retina class
 */
int StringImage::updateState(float time, float dt)
{
   int status = 0;

   status |= recvSynapticInput(NULL, NULL, 0);
   status |= Retina::updateState(time, dt);
   status |= outputState(time, dt);

   return status;
}

int StringImage::outputState(float time, float dt)
{
#ifdef DEBUG_OUTPUT
   int kex;
   printf("time==%f,", time);
   for (kex = 0; kex < 32; kex++) {
      if (string[kex] > 0) {
         printf("%c,", (char)string[kex]);
      }
      else {
         printf(" ,");
      }
   }
   printf("%c\n", (char)string[kex]);

   pvdata_t * data = getChannel(CHANNEL_EXC);
   for (kex = 8; kex < 15; kex++) {
      printf("%d,", (int)data[kex]);
   }
   printf("%d\n", (int)data[kex]);
#endif

   return 0;
}

} // namespace PV
