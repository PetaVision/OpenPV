/*
 * InitUniformRandomWeights.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#include "InitUniformRandomWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

InitUniformRandomWeights::InitUniformRandomWeights()
{
   initialize_base();
}

InitUniformRandomWeights::~InitUniformRandomWeights()
{
   free(rnd_state); rnd_state = NULL;
}

int InitUniformRandomWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitUniformRandomWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitUniformRandomWeightsParams(callingConn);
   return tempPtr;
}

int InitUniformRandomWeights::calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId,
      InitWeightsParams *weightParams) {
   InitUniformRandomWeightsParams *weightParamPtr = dynamic_cast<InitUniformRandomWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   const float wMinInit = weightParamPtr->getWMin();
   const float wMaxInit = weightParamPtr->getWMax();
   const float sparseFraction = weightParamPtr->getSparseFraction();

   uniformWeights(dataStart, wMinInit, wMaxInit, sparseFraction, weightParamPtr, patchIndex);

   //No longer setting delays in initWeights
   //weightParamPtr->getParentConn()->setDelay(arborId, arborId);

   return PV_SUCCESS; // return 1;
}

/**
 * uniformWeights() fills the full-size patch with random numbers, whether or not the patch is shrunken.
 */
int InitUniformRandomWeights::uniformWeights(
		/* PVPatch * wp */pvdata_t * dataStart, float minwgt, float maxwgt,
		float sparseFraction, InitUniformRandomWeightsParams *weightParamPtr, int patchIndex) {

   const int nxp = weightParamPtr->getnxPatch_tmp(); // wp->nx;
   const int nyp = weightParamPtr->getnyPatch_tmp(); // wp->ny;
   const int nfp = weightParamPtr->getnfPatch_tmp(); //wp->nf;

   const int sxp = weightParamPtr->getsx_tmp(); //wp->sx;
   const int syp = weightParamPtr->getsy_tmp(); //wp->sy;
   const int sfp = weightParamPtr->getsf_tmp(); //wp->sf;

   double p;
   if( maxwgt <= minwgt ) {
      if( maxwgt < minwgt ) {
         fprintf(stderr, "Warning: uniformWeights maximum less than minimum.  Changing max = %f to min value of %f\n", maxwgt, minwgt);
         maxwgt = minwgt;
      }
      p = 0;
   }
   else {
       p = (maxwgt - minwgt) / (1.0+(double) CL_RANDOM_MAX);
   }
   sparseFraction *= (1.0+(double) CL_RANDOM_MAX);

   uint4 rng = rnd_state[patchIndex];
   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            pvdata_t data = minwgt + (pvdata_t) (p * (double) rand_ul(&rng));
            if ((double) rand_ul(&rng) < sparseFraction) data = 0.0;
            dataStart[x * sxp + y * syp + f * sfp] = data;
         }
      }
   }

   return PV_SUCCESS;
}

/*
 * Each data patch has a unique cl_random random state.
 * For kernels, the data patch is seeded according to its patch index.
 * For non-kernels, the data patch is seeded according to the global index of its presynaptic neuron (which is in extended space)
 *     In MPI, in interior border regions, the same presynaptic neuron can have patches on more than one process.
 *     Patches on different processes with the same global pre-synaptic index will have the same seed and therefore
 *     will be identical.  Hence this implementation is independent of the MPI configuration.
 */
int InitUniformRandomWeights::initRNGs(HyPerConn * conn, bool isKernel) {
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

unsigned int InitUniformRandomWeights::rand_ul(uint4 * state) {
   // Generates a pseudo-random number in the range 0 to UINT_MAX (usually 2^32-1)
   *state = cl_random_get(*state);
   return state->s0;
}

} /* namespace PV */
