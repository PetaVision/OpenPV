/*
 * InitRandomWeights.cpp
 *
 *  Created on: Aug 21, 2013
 *      Author: pschultz
 */

#include "InitRandomWeights.hpp"

namespace PV {

InitRandomWeights::InitRandomWeights() {
   initialize_base();
}

int InitRandomWeights::initialize_base() {
   rnd_state = NULL;
   return PV_SUCCESS;
}

int InitRandomWeights::calcWeights(pvdata_t * dataStart, int dataPatchIndex, int arborId, InitWeightsParams *weightParams) {
   return randomWeights(dataStart, weightParams, &rnd_state[dataPatchIndex]); // RNG depends on dataPatchIndex but not on arborId.
}

/*
 * Each data patch has a unique cl_random random state.
 * For kernels, the data patch is seeded according to its patch index.
 * For non-kernels, the data patch is seeded according to the global index of its presynaptic neuron (which is in extended space)
 *     In MPI, in interior border regions, the same presynaptic neuron can have patches on more than one process.
 *     Patches on different processes with the same global pre-synaptic index will have the same seed and therefore
 *     will be identical.  Hence this implementation is independent of the MPI configuration.
 */
int InitRandomWeights::initRNGs(HyPerConn * conn, bool isKernel) {
   assert(rnd_state==NULL);
   int status = PV_SUCCESS;
   int numDataPatches = conn->getNumDataPatches();
   rnd_state = (uint4 *) malloc((size_t) (numDataPatches * sizeof(uint4)));
   int numGlobalRNGs = isKernel ? numDataPatches : conn->preSynapticLayer()->getNumGlobalExtended();
   unsigned long int seedBase = conn->getParent()->getObjectSeed(numGlobalRNGs);
   if (rnd_state==NULL) {
      fprintf(stderr, "InitUniformRandomWeights error in rank %d process: unable to allocate memory for random number state: %s", conn->getParent()->columnId(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   if (isKernel) {
      status = cl_random_init(rnd_state, numDataPatches, seedBase);
   }
   else {
      const PVLayerLoc * loc = conn->preSynapticLayer()->getLayerLoc();
      for (int y=0; y<loc->ny+2*loc->nb; y++) {
         unsigned long int kLineStartGlobal = (unsigned long int) kIndex(loc->kx0, loc->ky0+y, 0, loc->nxGlobal+2*loc->nb, loc->nyGlobal+2*loc->nb, loc->nf);
         int kLineStartLocal = kIndex(0, y, 0, loc->nx+2*loc->nb, loc->ny+2*loc->nb, loc->nf);
         if (cl_random_init(&rnd_state[kLineStartLocal], loc->nx+2*loc->nb, seedBase + kLineStartGlobal)!=PV_SUCCESS) status = PV_FAILURE;
      }
   }
   return status;
}

InitRandomWeights::~InitRandomWeights() {
   free(rnd_state); rnd_state = NULL;
}

} /* namespace PV */
