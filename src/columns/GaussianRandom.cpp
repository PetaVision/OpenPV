/*
 * GaussianRandom.cpp
 *
 *  Created on: Aug 23, 2013
 *      Author: pschultz
 */

#include "GaussianRandom.hpp"

namespace PV {

GaussianRandom::GaussianRandom(int count) {
   initialize_base();
   initializeFromCount((unsigned int) count);
}

GaussianRandom::GaussianRandom(const PVLayerLoc * locptr, bool isExtended) {
   initialize_base();
   initializeFromLoc(locptr, isExtended);
}

GaussianRandom::GaussianRandom() {
   initialize_base();
}

int GaussianRandom::initialize_base() {
   return PV_SUCCESS;
}

int GaussianRandom::initializeGaussian(){
   int status = PV_SUCCESS;
   heldValues.assign(rngArray.size(), {false, 0.0});
   return status;
}

int GaussianRandom::initializeFromCount(unsigned int count) {
   int status = Random::initializeFromCount(count);
   if (status == PV_SUCCESS) {
      status = initializeGaussian();
   }
   return status;
}

int GaussianRandom::initializeFromLoc(const PVLayerLoc* locptr, bool isExtended) {
   int status = Random::initializeFromLoc(locptr, isExtended);
   if(status == PV_SUCCESS){
      status = initializeGaussian();
   }
   return status;
}

// gaussianDist uses the Box-Muller transformation, using code based on the routine at
// <http://code.google.com/p/bzflags/source/browse/trunk/inc/boxmuller.c?spec=svn113&r=113>.
//
// /* boxmuller.c           Implements the Polar form of the Box-Muller
//                          Transformation
//
//                       (c) Copyright 1994, Everett F. Carter Jr.
//                           Permission is granted by the author to use
//                           this software for any application provided this
//                           copyright notice is preserved.
//
// */
//
// #include <math.h>
// #include <cstdlib>
//
// float ranf();
//
// float ranf() {
//     int randNum = rand();
//     float result = randNum / RAND_MAX;
//     return result;
// }/* ranf() is uniform in 0..1 */
//
//
// float box_muller(float m, float s)      /* normal random variate generator */
// {                                       /* mean m, standard deviation s */
//         float x1, x2, w, y1;
//         static float y2;
//         static int use_last = 0;
//
//         if (use_last)                   /* use value from previous call */
//         {
//                 y1 = y2;
//                 use_last = 0;
//         }
//         else
//         {
//                 do {
//                         x1 = 2.0 * ranf() - 1.0;
//                         x2 = 2.0 * ranf() - 1.0;
//                         w = x1 * x1 + x2 * x2;
//                 } while ( w >= 1.0 );
//
//                 w = sqrt( (-2.0 * log( w ) ) / w );
//                 y1 = x1 * w;
//                 y2 = x2 * w;
//                 use_last = 1;
//         }
//
//         return( m + y1 * s );
// }
float GaussianRandom::gaussianDist(int localIndex) {
   float x1, x2, y;
   struct box_muller_data bmdata = heldValues[localIndex];
   if (bmdata.hasHeldValue) {
      y = bmdata.heldValue;
   }
   else {
      float w;
      do {
         x1 = 2.0 * uniformRandom(localIndex) - 1.0;
         x2 = 2.0 * uniformRandom(localIndex) - 1.0;
         w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );

      w = sqrt( (-2.0 * log( w ) ) / w );
      y = x1 * w;
      bmdata.heldValue = x2 * w;
   }
   bmdata.hasHeldValue = !bmdata.hasHeldValue;

   return y;
}


GaussianRandom::~GaussianRandom() {
}

} /* namespace PV */
