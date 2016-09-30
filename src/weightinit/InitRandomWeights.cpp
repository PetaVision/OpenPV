/*
 * InitRandomWeights.cpp
 *
 *  Created on: Aug 21, 2013
 *      Author: pschultz
 */

#include "InitRandomWeights.hpp"

namespace PV {

InitRandomWeights::InitRandomWeights() { initialize_base(); }

int InitRandomWeights::initialize_base() {
   randState = NULL;
   return PV_SUCCESS;
}

int InitRandomWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

int InitRandomWeights::calcWeights(pvdata_t *dataStart, int dataPatchIndex, int arborId) {
   return randomWeights(
         dataStart,
         weightParams,
         dataPatchIndex); // RNG depends on dataPatchIndex but not on arborId.
}

/*
 * Each data patch has a unique random state in the randState object.
 * For kernels, the data patch is seeded according to its patch index.
 * For non-kernels, the data patch is seeded according to the global index of its presynaptic neuron
 * (which is in extended space)
 *     In MPI, in interior border regions, the same presynaptic neuron can have patches on more than
 * one process.
 *     Patches on different processes with the same global pre-synaptic index will have the same
 * seed and therefore
 *     will be identical.  Hence this implementation is independent of the MPI configuration.
 */
int InitRandomWeights::initRNGs(bool isKernel) {
   assert(randState == NULL);
   int status = PV_SUCCESS;
   if (isKernel) {
      randState = new Random(callingConn->getNumDataPatches());
   } else {
      randState = new Random(callingConn->preSynapticLayer()->getLayerLoc(), true /*isExtended*/);
   }
   if (randState == NULL) {
      pvError().printf(
            "InitRandomWeights error in rank %d process: unable to create object of class "
            "Random.\n",
            callingConn->getParent()->columnId());
   }
   return status;
}

InitRandomWeights::~InitRandomWeights() {
   delete randState;
   randState = NULL;
}

} /* namespace PV */
