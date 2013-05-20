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
#ifdef PV_USE_MPI
   mpiReductionBuffer = NULL;
#endif // PV_USE_MPI
   int status = KernelConn::initialize(name, hc, pre, post, NULL, weightInit);
   delete weightInit;
   return status;
}

int IdentConn::setParams(PVParams * inputParams) {
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   if( preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny || preLoc->nf != postLoc->nf ) {
      if (parent->columnId()==0) {
         fprintf( stderr,
                  "IdentConn Error: %s and %s do not have the same dimensions.\n Dims: %dx%dx%d vs. %dx%dx%d\n",
                  pre->getName(),post->getName(),preLoc->nx,preLoc->ny,preLoc->nf,postLoc->nx,postLoc->ny,postLoc->nf);
      }
      exit(EXIT_FAILURE);
   }
   int status = KernelConn::setParams(inputParams);

   return status;
}

void IdentConn::readNumAxonalArbors(PVParams * params) {
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

void IdentConn::readCombine_dW_with_W_flag(PVParams * params) {
   return;
}

int IdentConn::readPatchSize(PVParams * params) {
   nxp = 1;
   nyp = 1;
   nxpShrunken = 1;
   nypShrunken = 1;
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
