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

// The most basic public constructor.  For example, if numBlocks=4, blockLength=6, numGlobalBlocks=10, globalBlockLength=8, and startIndex=7,
// the RNG array will have 4*6=24 elements, and their seeds will be a base seed plus the following values
//    7   8   9  10  11  12
//   15  16  17  18  19  20
//   23  24  25  26  27  28
//   31  32  33  34  35  36.
// The base seed is obtained by calling hc->getObjectSeed(10*8).
Random::Random(HyPerCol * hc, unsigned int numBlocks, unsigned int blockLength, unsigned int numGlobalBlocks, unsigned int globalBlockLength, unsigned int startIndex) {
   initialize_base();
   initialize(hc, numBlocks, blockLength, numGlobalBlocks, globalBlockLength, startIndex);
}

// N independent random number generators, all processes have the same N seeds.
Random::Random(HyPerCol * hc, int count) {
   initialize_base();
   initialize(hc, 1U, (unsigned int) count, 1U, (unsigned int) count, 0U);
}

// Each neuron in a layer has its own RNG.  locptr defines the size of the layer.
// isExtended tells whether to consider getNumGlobalNeurons() or getNumGlobalExtended() neurons.
// The seed of each RNG is determined by *global* index; this way the initial state of the
// random number does not depend on the MPI configuration.
Random::Random(HyPerCol * hc, const PVLayerLoc * locptr, bool isExtended) {
   initialize_base();
   unsigned int numBlocks, blockLength, numGlobalBlocks, nxGlobal, globalBlockLength, startIndex;
   defineBlocksFromPVLayerLoc(locptr, isExtended, &numBlocks, &blockLength, &numGlobalBlocks, &globalBlockLength, &startIndex);

   initialize(hc, numBlocks, blockLength, numGlobalBlocks, globalBlockLength, startIndex);
}

int Random::initialize_base() {
   parentHyPerCol = NULL;
   numBlocks = 0U;
   blockLength = 0U;
   numGlobalBlocks = 0U;
   globalBlockLength = 0U;
   startIndex = 0U;
   rngArray = NULL;
   rngArraySize = 0UL;
   return PV_SUCCESS;
}

int Random::initialize(HyPerCol * hc, unsigned int numBlocks, unsigned int blockLength, unsigned int numGlobalBlocks, unsigned int globalBlockLength, unsigned int startIndex) {
   int status = PV_SUCCESS;
   parentHyPerCol = hc;
   this->numBlocks = numBlocks;
   this->blockLength = blockLength;
   this->numGlobalBlocks = numGlobalBlocks;
   this->globalBlockLength = globalBlockLength;
   this->startIndex = startIndex;
   rngArraySize = numBlocks*blockLength;
   rngArray = (uint4 *) malloc(rngArraySize*sizeof(uint4));
   if (rngArray==NULL) {
      fprintf(stderr, "Random::initialize error: rank %d process unable to allocate memory for %lu RNGs.\n", hc->columnId(), rngArraySize);
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      int globalSize = (int) (numGlobalBlocks*globalBlockLength);
      unsigned int seedBase = hc->getObjectSeed(globalSize);
      seedBase += startIndex;
      unsigned int localIndex = 0;
      for (int y=0; y<numBlocks; y++) {
         cl_random_init(&rngArray[localIndex], (size_t) blockLength, seedBase);
         localIndex += blockLength;
         seedBase += globalBlockLength;
      }
   }
   return status;
}
int Random::defineBlocksFromPVLayerLoc(PVLayerLoc const * loc, bool isExtended, unsigned int * numBlocks, unsigned int * blockLength, unsigned int * numGlobalBlocks, unsigned int * globalBlockLength, unsigned int * startIndex) {
   PVHalo halo;
   if (isExtended) {
      memcpy(&halo, &(loc->halo), sizeof(halo));
   }
   else {
      halo.lt = 0;
      halo.rt = 0;
      halo.dn = 0;
      halo.up = 0;
   }
   *numBlocks = loc->ny+halo.dn + halo.up;
   *blockLength =  (loc->nx+halo.lt+halo.rt)*loc->nf;
   *numGlobalBlocks = loc->nyGlobal+halo.dn+halo.up;
   int nxGlobal = loc->nxGlobal+halo.lt+halo.rt;
   *globalBlockLength = nxGlobal*loc->nf;
   *startIndex = kIndex(loc->kx0, loc->ky0, 0, nxGlobal, (int) *numGlobalBlocks, loc->nf);
   return PV_SUCCESS;
}

float Random::uniformRandom(int localIndex) {
   rngArray[localIndex] = cl_random_get(rngArray[localIndex]);
   return rngArray[localIndex].s0/(float) randomUIntMax();
}

unsigned int Random::randomUInt(int localIndex) {
   rngArray[localIndex] = cl_random_get(rngArray[localIndex]);
   return rngArray[localIndex].s0;
}

Random::~Random() {
   free(rngArray); rngArray = NULL;
}

} /* namespace PV */
