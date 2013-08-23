/*
 * Random.cpp
 *
 *  Created on: Aug 23, 2013
 *      Author: pschultz
 */

#include "Random.hpp"

namespace PV {

Random::Random() {
   // Default constructor is called only by derived class constructors.
   // Derived classes should call Random::initialize() themselves.
   initialize_base();
}

Random::Random(HyPerCol * hc, const PVLayerLoc * locptr, bool isExtended) {
   initialize_base();
   initialize(hc, locptr, isExtended);
}

int Random::initialize_base() {
   parentHyPerCol = NULL;
   memcpy(&loc, 0, sizeof(PVLayerLoc));
   rngArray = NULL;
   rngArraySize = 0UL;
   return PV_SUCCESS;
}

int Random::initialize(HyPerCol * hc, const PVLayerLoc * locptr, bool isExtended) {
   int status = PV_SUCCESS;
   parentHyPerCol = hc;
   memcpy(&loc, locptr, sizeof(PVLayerLoc));
   loc.nb = isExtended ? loc.nb : 0;
   int yStrideLocal = (loc.nx+2*loc.nb)*loc.nf;
   int yStrideGlobal = (loc.nxGlobal+2*loc.nb)*loc.nf;
   int numSeeds = (loc.nyGlobal+2*loc.nb)*yStrideLocal;
   unsigned int seedBase = hc->getObjectSeed(numSeeds);
   rngArraySize = (size_t) ((loc.ny+2*loc.nb)*yStrideLocal);
   rngArray = (uint4 *) malloc(rngArraySize*sizeof(uint4));
   if (rngArray==NULL) {
      fprintf(stderr, "Random::initialize error: rank %d process unable to allocate memory for %lu RNGs.\n", hc->columnId(), rngArraySize);
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      int ny = loc.ny + 2*loc.nb; // Recall that nb=0 if isExtended is false
      int localIndex = 0;
      int globalIndex = kIndex(loc.kx0, loc.ky0, 0, loc.nxGlobal+2*loc.nb, loc.nyGlobal+2*loc.nb, loc.nf);
      seedBase += (unsigned int) globalIndex;
      for (int y=0; y<ny; y++) {
         cl_random_init(&rngArray[localIndex], (size_t) yStrideLocal, seedBase);
         localIndex += yStrideLocal;
         seedBase += (unsigned int) yStrideGlobal;
      }
   }
   return status;
}

float Random::uniformRandom(int localIndex) {
   rngArray[localIndex] = cl_random_get(rngArray[localIndex]);
   return rngArray[localIndex].s0/(float) cl_random_max();
}

Random::~Random() {
}

} /* namespace PV */
