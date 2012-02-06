/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"
#include "../weightinit/InitIdentWeights.hpp"

namespace PV {

IdentConn::IdentConn() {
    initialize_base();
}

IdentConn::IdentConn(const char * name, HyPerCol *hc,
        HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL);
}  // end of IdentConn::IdentConn(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, ChannelType)

int IdentConn::initialize_base() {
   // no IdentConn-specific data members to initialize
   return PV_SUCCESS;
}  // end of IdentConn::initialize_base()

int IdentConn::initialize( const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename ) {
   InitIdentWeights * weightInit = new InitIdentWeights;
   if( weightInit == NULL ) {
      fprintf(stderr, "IdentConn \"%s\": Rank %d process unable to create InitIdentWeights object.  Exiting.\n", name, hc->icCommunicator()->commRank());
      exit(EXIT_FAILURE);
   }
   symmetrizeWeightsFlag = false; // The data members set here should not be used by IdentConn.
   weightUpdateTime = -1;         // Give them safe values nonetheless, as a precaution.
#ifdef PV_USE_MPI
   mpiReductionBuffer = NULL;
#endif PV_USE_MPI
   int status = HyPerConn::initialize(name, hc, pre, post, channel, NULL, weightInit);
   delete weightInit;
   return status;
}

int IdentConn::setParams(PVParams * inputParams) {
   numAxonalArborLists=1;
   plasticityFlag = false;
   stochasticReleaseFlag = false;
   writeCompressedWeights = true;
   return PV_SUCCESS;
}

int IdentConn::setPatchSize(const char * filename) {
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   if( preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny || preLoc->nf != postLoc->nf ) {
      fprintf( stderr,
               "IdentConn Error: %s and %s do not have the same dimensions\n",
               pre->getName(),post->getName() );
      exit(EXIT_FAILURE);
   }
   nxp = 1;
   nyp = 1;
   nfp = preLoc->nf;

   return PV_SUCCESS;
}  // end of IdentConn::setPatchSize(const char *)

#ifdef OBSOLETE // This method has been moved to InitIdentWeights.  To initialize IdentConns set the param "weightInitType" to "InitIdentWeight" in the params file
PVPatch ** IdentConn::initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename) {
    int numKernels = numDataPatches(0);
    for( int k=0; k < numKernels; k++ ) {
        PVPatch * kp = getKernelPatch(k);
        assert(kp->nf == numKernels);
        for( int l=0; l < kp->nf; l++ ) {
            kp->data[l] = l==k;
        }
    }
    return patches;
}  // end of IdentConn::initializeWeights(PVPatch **, int, const char *)
#endif // OBSOLETE

int IdentConn::initNormalize() {
   normalize_flag = false; // Make sure that updateState doesn't call normalizeWeights
   return PV_SUCCESS;
}

int IdentConn::initShrinkPatches() {
   shrinkPatches_flag = false;
   return PV_SUCCESS;
}

}  // end of namespace PV block
