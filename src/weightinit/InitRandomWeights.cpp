/*
 * InitRandomWeights.cpp
 *
 *  Created on: Aug 21, 2013
 *      Author: pschultz
 */

#include "InitRandomWeights.hpp"

namespace PV {

InitRandomWeights::InitRandomWeights() {}

InitRandomWeights::~InitRandomWeights() {
   delete mRandState;
   mRandState = nullptr;
}

int InitRandomWeights::initialize(char const *name, HyPerCol *hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

void InitRandomWeights::calcWeights(int dataPatchIndex, int arborId) {
   randomWeights(mWeights->getDataFromDataIndex(arborId, dataPatchIndex), dataPatchIndex);
   // RNG depends on dataPatchIndex but not on arborId.
}

/*
 * Each data patch has a unique random state in the mRandState object.
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
   assert(mRandState == nullptr);
   int status = PV_SUCCESS;
   if (isKernel) {
      mRandState = new Random(mWeights->getNumDataPatches());
   }
   else {
      mRandState = new Random(&mWeights->getGeometry()->getPreLoc(), true /*isExtended*/);
   }
   if (mRandState == nullptr) {
      Fatal().printf(
            "InitRandomWeights error in rank %d process: unable to create object of class "
            "Random.\n",
            parent->getCommunicator()->globalCommRank());
   }
   return status;
}

} /* namespace PV */
