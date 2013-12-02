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
      const char * pre_layer_name, const char * post_layer_name) {
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, NULL);
}  // end of IdentConn::IdentConn(const char *, HyPerCol *, HyPerLayer *, HyPerLayer *, ChannelType)

int IdentConn::initialize_base() {
   // no IdentConn-specific data members to initialize
   return PV_SUCCESS;
}  // end of IdentConn::initialize_base()

int IdentConn::initialize( const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name, const char * filename ) {
   InitIdentWeights * weightInit = new InitIdentWeights;
   if( weightInit == NULL ) {
      fprintf(stderr, "IdentConn \"%s\": Rank %d process unable to create InitIdentWeights object.  Exiting.\n", name, hc->icCommunicator()->commRank());
      exit(EXIT_FAILURE);
   }
#ifdef PV_USE_MPI
   mpiReductionBuffer = NULL;
#endif // PV_USE_MPI
   int status = KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, NULL, weightInit);
   return status;
}

int IdentConn::setParams(PVParams * inputParams) {
   int status = KernelConn::setParams(inputParams);

   return status;
}

void IdentConn::readNumAxonalArbors(PVParams * params) {
   numAxonalArborLists=1;
   handleUnnecessaryIntParameter("numAxonalArbors", numAxonalArborLists);
}

void IdentConn::readPlasticityFlag(PVParams * params) {
   plasticityFlag = false;
   handleUnnecessaryIntParameter("plasticityFlag", plasticityFlag);
}

void IdentConn::readKeepKernelsSynchronized(PVParams * params) {
   keepKernelsSynchronized_flag = true;
   handleUnnecessaryIntParameter("keepKernelsSynchronized", keepKernelsSynchronized_flag);
}

void IdentConn::readWeightUpdatePeriod(PVParams * params) {
   weightUpdatePeriod = 1.0f;
   handleUnnecessaryFloatingPointParameter("weightUpdatePeriod", weightUpdatePeriod);
}

void IdentConn::readInitialWeightUpdateTime(PVParams * params) {
   weightUpdateTime = 0.0f;
   handleUnnecessaryFloatingPointParameter("initialWeightUpdateTime", weightUpdateTime);
}

void IdentConn::readPvpatchAccumulateType(PVParams * params) {
   pvpatchAccumulateType = ACCUMULATE_CONVOLVE;
   handleUnnecessaryParameterString("pvpatchAccumulateType", "convolve", true/*case insensitive*/);
}

void IdentConn::readPreActivityIsNotRate(PVParams * params) {
   preActivityIsNotRate = false;
   handleUnnecessaryIntParameter("preActivityIsNotRate", preActivityIsNotRate);
}

void IdentConn::readWriteCompressedWeights(PVParams * params) {
   writeCompressedWeights = true;
   handleUnnecessaryIntParameter("writeCompressedWeights", writeCompressedWeights);
}

void IdentConn::readWriteCompressedCheckpoints(PVParams * params) {
   writeCompressedCheckpoints = true;
   handleUnnecessaryIntParameter("writeCompressedCheckpoints", writeCompressedCheckpoints);
}

void IdentConn::readSelfFlag(PVParams * params) {
   selfFlag = false;
   handleUnnecessaryIntParameter("selfFlag", selfFlag);
}

void IdentConn::readCombine_dW_with_W_flag(PVParams * params) {
   assert(plasticityFlag==false);
   // readCombine_dW_with_W_flag only used if when plasticityFlag is true, which it never is for IdentConn
   return;
}

int IdentConn::readPatchSize(PVParams * params) {
   nxp = 1;
   nyp = 1;
   nxpShrunken = 1;
   nypShrunken = 1;
   handleUnnecessaryIntParameter("nxp", nxp);
   handleUnnecessaryIntParameter("nyp", nyp);
   handleUnnecessaryIntParameter("nxpShrunken", nxpShrunken);
   handleUnnecessaryIntParameter("nypShrunken", nypShrunken);
   return PV_SUCCESS;
}

int IdentConn::readNfp(PVParams * params) {
   nfp = -1;
   // nfp is set in HyPerConn::communicateInitInfo(), where it is copied from postsynaptic layer's nf.
   warnDefaultNfp = false;
   return PV_SUCCESS;
}

void IdentConn::readShrinkPatches(PVParams * params) {
   shrinkPatches_flag = false;
   handleUnnecessaryIntParameter("shrinkPatches", shrinkPatches_flag);
}

void IdentConn::readUpdateGSynFromPostPerspective(PVParams * params){
   updateGSynFromPostPerspective = false;
   handleUnnecessaryIntParameter("updateGSynFromPostPerspective", updateGSynFromPostPerspective);
}

int IdentConn::communicateInitInfo() {
   int status = KernelConn::communicateInitInfo();
   assert(pre && post);
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   if( preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny || preLoc->nf != postLoc->nf ) {
      if (parent->columnId()==0) {
         fprintf( stderr,
                  "IdentConn \"%s\" Error: %s and %s do not have the same dimensions.\n Dims: %dx%dx%d vs. %dx%dx%d\n",
                  name, preLayerName,postLayerName,preLoc->nx,preLoc->ny,preLoc->nf,postLoc->nx,postLoc->ny,postLoc->nf);
      }
      exit(EXIT_FAILURE);
   }
   handleUnnecessaryIntParameter("nfp", nfp); // nfp is set during call to KernelConn::communicateInitInfo, so don't check for unnecessary int parameter until after that.
   return status;
}

int IdentConn::initNormalize() {
   // normalize_flag = false; // Make sure that updateState doesn't call normalizeWeights
   return PV_SUCCESS;
}

}  // end of namespace PV block
