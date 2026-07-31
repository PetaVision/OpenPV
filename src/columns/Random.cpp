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
}

// N independent random number generators, all processes have the same N seeds.
Random::Random(long count) {
   initializeFromCount(count);
}

// Each neuron in a layer has its own RNG.  locptr defines the geometry of the
// layer.
// isExtended tells whether to consider getNumGlobalNeurons() or
// getNumGlobalExtended() neurons.
// The seed of each RNG is determined by *global* index; this way the initial
// state of the
// random number does not depend on the MPI configuration.
Random::Random(const PVLayerLoc *locptr, bool isExtended) {
   initializeFromLoc(locptr, isExtended);
}

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
   int nxExt     = locptr->nx + halo.lt + halo.rt;
   int nyExt     = locptr->ny + halo.up + halo.dn;
   int nf        = locptr->nf;
   int nbatch    = locptr->nbatch;
   long rngCount = (long)nxExt * (long)nyExt * (long)nf * (long)nbatch;
   // Calculate global size
   int nxGlobalExt  = locptr->nxGlobal + halo.lt + halo.rt;
   int nyGlobalExt  = locptr->nyGlobal + halo.up + halo.dn;
   int nbatchGlobal = locptr->nbatchGlobal;
   // Allocate buffer to store rngArraySize
   mRNG.resize(rngCount);
   if (status == PV_SUCCESS) {
      long numTotalSeeds     = (long)nxGlobalExt * (long)nyGlobalExt * (long)nf * (long)nbatchGlobal;
      unsigned long seedBase = RandomSeed::instance()->allocate(numTotalSeeds);
      long sb                = (long)nxExt * (long)nyExt * (long)nf;
      long sy                = (long)nxExt * (long)nf;
      long sbGlobal          = (long)nxGlobalExt * (long)nyGlobalExt * (long)nf;
      long syGlobal          = (long)nxGlobalExt * (long)nf;

      // Only thing that is continuous in memory is nx and nf, so loop over batch
      // and y
      for (int kb = 0; kb < nbatch; kb++) {
         for (int ky = 0; ky < nyExt; ky++) {
            // Calculate start index into local RNG
            long localExtStart = kb * sb + ky * sy;
            // Calculate offset of the seedBase
            long globalExtStart =
                  (kb + locptr->kb0) * sbGlobal + (ky + locptr->ky0) * syGlobal + locptr->kx0 * nf;
            std::size_t count = static_cast<std::size_t>(nxExt) * static_cast<std::size_t>(nf);
            cl_random_init(&(mRNG[localExtStart]), count, seedBase + (unsigned long)globalExtStart);
         }
      }
   }
   return status;
}

int Random::initializeFromCount(long count) {
   int status = PV_SUCCESS;
   mRNG.resize(count);
   if (status == PV_SUCCESS) {
      unsigned long seedBase = RandomSeed::instance()->allocate(static_cast<unsigned long>(count));
      cl_random_init(mRNG.data(), (size_t)count, seedBase);
   }
   return status;
}

float Random::uniformRandom(long localIndex) {
   mRNG[localIndex] = cl_random_get(mRNG[localIndex]);
   return static_cast<float>(mRNG[localIndex].s0) / static_cast<float>(randomUIntMax());
}

unsigned int Random::randomUInt(long localIndex) {
   mRNG[localIndex] = cl_random_get(mRNG[localIndex]);
   return mRNG[localIndex].s0;
}

Random::~Random() {}

} /* namespace PV */
