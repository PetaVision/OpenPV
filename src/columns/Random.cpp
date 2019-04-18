/*
 * Random.cpp
 *
 *  Created on: Aug 23, 2013
 *      Author: pschultz
 */

#include "Random.hpp"
#include "columns/RandomSeed.hpp"
#include "utils/PVLog.hpp"

namespace PV {

Random::Random() {
   // Default constructor is called only by derived class constructors.
   // Derived classes should call Random::initialize() themselves.
   initialize_base();
}

// N independent random number generators, all processes have the same N seeds.
Random::Random(int count) {
   initialize_base();
   initializeFromCount((unsigned int)count);
}

// Each neuron in a layer has its own RNG.  locptr defines the geometry of the
// layer.
// isExtended tells whether to consider getNumGlobalNeurons() or
// getNumGlobalExtended() neurons.
// The seed of each RNG is determined by *global* index; this way the initial
// state of the
// random number does not depend on the MPI configuration.
Random::Random(const PVLayerLoc *locptr, bool isExtended) {
   initialize_base();
   initializeFromLoc(locptr, isExtended);
}

int Random::initialize_base() { return PV_SUCCESS; }

// Initialize with repsect to nbatch, nx, ny, nf in extended space
int Random::initializeFromLoc(const PVLayerLoc *locptr, bool isExtended) {
   int status = PV_SUCCESS;

   PVHalo halo;
   if (isExtended) {
      halo = locptr->halo;
   }
   else {
      halo.lt = 0;
      halo.rt = 0;
      halo.dn = 0;
      halo.up = 0;
   }
   int nxExt    = locptr->nx + halo.lt + halo.rt;
   int nyExt    = locptr->ny + halo.up + halo.dn;
   int nf       = locptr->nf;
   int nbatch   = locptr->nbatch;
   int rngCount = nxExt * nyExt * nf * nbatch;
   // Calculate global size
   int nxGlobalExt  = locptr->nxGlobal + halo.lt + halo.rt;
   int nyGlobalExt  = locptr->nyGlobal + halo.up + halo.dn;
   int nbatchGlobal = locptr->nbatchGlobal;
   // Allocate buffer to store rngArraySize
   rngArray.resize(rngCount);
   if (status == PV_SUCCESS) {
      int numTotalSeeds     = nxGlobalExt * nyGlobalExt * nf * nbatchGlobal;
      unsigned int seedBase = RandomSeed::instance()->allocate(numTotalSeeds);
      int sb                = nxExt * nyExt * nf;
      int sy                = nxExt * nf;
      int sbGlobal          = nxGlobalExt * nyGlobalExt * nf;
      int syGlobal          = nxGlobalExt * nf;

      // Only thing that is continuous in memory is nx and ny, so loop over batch
      // and y
      for (int kb = 0; kb < nbatch; kb++) {
         for (int ky = 0; ky < nyExt; ky++) {
            // Calculate start index into local rngArray
            int localExtStart = kb * sb + ky * sy;
            // Calculate offset of the seedBase
            int globalExtStart =
                  (kb + locptr->kb0) * sbGlobal + (ky + locptr->ky0) * syGlobal + locptr->kx0;
            size_t count = nxExt * nf;
            cl_random_init(&(rngArray[localExtStart]), count, seedBase + globalExtStart);
         }
      }
   }
   return status;
}

int Random::initializeFromCount(int count) {
   int status = PV_SUCCESS;
   rngArray.resize(count);
   if (status == PV_SUCCESS) {
      unsigned int seedBase = RandomSeed::instance()->allocate(count);
      cl_random_init(rngArray.data(), (size_t)count, seedBase);
   }
   return status;
}

float Random::uniformRandom(int localIndex) {
   rngArray[localIndex] = cl_random_get(rngArray[localIndex]);
   return rngArray[localIndex].s0 / (float)randomUIntMax();
}

unsigned int Random::randomUInt(int localIndex) {
   rngArray[localIndex] = cl_random_get(rngArray[localIndex]);
   return rngArray[localIndex].s0;
}

Random::~Random() {}

} /* namespace PV */
