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
        HyPerLayer * pre, HyPerLayer * post) {
   initialize_base();
   initialize(name, hc, pre, post, NULL);
}  // end of IdentConn::IdentConn(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, ChannelType)

int IdentConn::initialize_base() {
   // no IdentConn-specific data members to initialize
   return PV_SUCCESS;
}  // end of IdentConn::initialize_base()

int IdentConn::initialize( const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, const char * filename ) {
   InitIdentWeights * weightInit = new InitIdentWeights;
   if( weightInit == NULL ) {
      fprintf(stderr, "IdentConn \"%s\": Rank %d process unable to create InitIdentWeights object.  Exiting.\n", name, hc->icCommunicator()->commRank());
      exit(EXIT_FAILURE);
   }
   symmetrizeWeightsFlag = false; // The data members set here should not be used by IdentConn.
   weightUpdateTime = -1;         // Give them safe values nonetheless, as a precaution.
#ifdef PV_USE_MPI
   mpiReductionBuffer = NULL;
#endif // PV_USE_MPI
   int status = HyPerConn::initialize(name, hc, pre, post, NULL, weightInit);
#ifdef PV_USE_OPENCL
   //don't support GPU acceleration in kernelconn yet
   ignoreGPUflag=true;
   //tell the receiving layer to copy gsyn to the gpu, because kernelconn won't be calculating it
   //post->copyChannelToDevice();
#endif
   initPatchToDataLUT();
   delete weightInit;
   return status;
}

int IdentConn::setParams(PVParams * inputParams) {
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   if( preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny || preLoc->nf != postLoc->nf ) {
      if (parent->columnId()==0) {
         fprintf( stderr,
                  "IdentConn Error: %s and %s do not have the same dimensions\n",
                  pre->getName(),post->getName() );
      }
      exit(EXIT_FAILURE);
   }
   int status = KernelConn::setParams(inputParams);

   return status;
}

void IdentConn::readNumAxonalArborLists(PVParams * params) {
   numAxonalArborLists=1;
}

void IdentConn::readPlasticityFlag(PVParams * params) {
   plasticityFlag = false;
}

void IdentConn::readKeepKernelsSynchronized(PVParams * params) {
   keepKernelsSynchronized_flag = true;
}

void IdentConn::readWeightUpdatePeriod(PVParams * params) {
   weightUpdatePeriod = 1.0f;
}

void IdentConn::readInitialWeightUpdateTime(PVParams * params) {
   weightUpdateTime = 0.0f;
}

void IdentConn::readStochasticReleaseFlag(PVParams * params) {
   stochasticReleaseFlag = false;
}

void IdentConn::readPreActivityIsNotRate(PVParams * params) {
   preActivityIsNotRate = false;
}

void IdentConn::readWriteCompressedWeights(PVParams * params) {
   writeCompressedWeights = true;
}

void IdentConn::readWriteCompressedCheckpoints(PVParams * params) {
   writeCompressedCheckpoints = true;
}

void IdentConn::readSelfFlag(PVParams * params) {
   assert(pre!=NULL && post!=NULL);
   selfFlag = pre==post;
}

int IdentConn::readNxp(PVParams * params) {
   nxp = 1;
   return PV_SUCCESS;
}

int IdentConn::readNyp(PVParams * params) {
   nyp = 1;
   return PV_SUCCESS;
}

int IdentConn::readNfp(PVParams * params) {
   nfp = pre->getLayerLoc()->nf;
   assert(nfp==post->getLayerLoc()->nf);
   return PV_SUCCESS;
}

int IdentConn::setPatchSize(const char * filename) {
   return PV_SUCCESS;
}  // end of IdentConn::setPatchSize(const char *)

int IdentConn::initNormalize() {
   normalize_flag = false; // Make sure that updateState doesn't call normalizeWeights
   return PV_SUCCESS;
}

void IdentConn::readShrinkPatches(PVParams * params) {
   shrinkPatches_flag = false;
}

}  // end of namespace PV block
