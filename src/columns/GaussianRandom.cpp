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

float GaussianRandom::gaussianDist(int localIndex) {
   float x1, x2, y;
   struct box_muller_data bmdata = heldValues[localIndex];
   if (bmdata.hasHeldValue) {
      y = bmdata.heldValue;
   }
   else {
      float w;
      do {
         x1 = 2.0f * uniformRandom(localIndex) - 1.0f;
         x2 = 2.0f * uniformRandom(localIndex) - 1.0f;
         w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0f );

      w = sqrtf( (-2.0f * logf( w ) ) / w );
      y = x1 * w;
      bmdata.heldValue = x2 * w;
   }
   bmdata.hasHeldValue = !bmdata.hasHeldValue;

   return y;
}


GaussianRandom::~GaussianRandom() {
}

} /* namespace PV */
