/*
 * PoolingConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "PoolingConn.hpp"
#include <cstring>

namespace PV {

PoolingConn::PoolingConn(){
   initialize_base();
}

PoolingConn::PoolingConn(const char * name, HyPerCol * hc) : HyPerConn()
{
   initialize_base();
   initialize(name, hc, NULL, NULL);
}

PoolingConn::~PoolingConn() {
}

int PoolingConn::initialize_base() {
   return PV_SUCCESS;
}

int PoolingConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   return status;
}

void PoolingConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag");
   }
}

void PoolingConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

int PoolingConn::initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   // It is okay for either of weightInitializer or weightNormalizer to be null at this point, either because we're in a subclass that doesn't need it, or because we are allowing for
   // backward compatibility.
   // The two lines needs to be before the call to BaseConnection::initialize, because that function calls ioParamsFillGroup,
   // which will call ioParam_weightInitType and ioParam_normalizeMethod, which for reasons of backward compatibility
   // will create the initializer and normalizer if those member variables are null.
   this->weightInitializer = weightInitializer;
   this->normalizer = weightNormalizer;

   int status = BaseConnection::initialize(name, hc); // BaseConnection should *NOT* take weightInitializer or weightNormalizer as arguments, as it does not know about InitWeights or NormalizeBase

   assert(parent);
   PVParams * inputParams = parent->parameters();

   //set accumulateFunctionPointer
   assert(!inputParams->presentAndNotBeenRead(name, "pvpatchAccumulateType"));
   switch (pvpatchAccumulateType) {
   case ACCUMULATE_CONVOLVE:
      std::cout << "ACCUMULATE_CONVOLVE not allowed in pooling conn\n";
      exit(-1);
      break;
   case ACCUMULATE_STOCHASTIC:
      std::cout << "ACCUMULATE_STOCASTIC not allowed in pooling conn\n";
      exit(-1);
      break;
   case ACCUMULATE_MAXPOOLING:
      accumulateFunctionPointer = &pvpatch_max_pooling;
      accumulateFunctionFromPostPointer = &pvpatch_max_pooling_from_post;
      break;
   case ACCUMULATE_SUMPOOLING:
      accumulateFunctionPointer = &pvpatch_sum_pooling;
      accumulateFunctionFromPostPointer = &pvpatch_sumpooling_from_post;
      break;
   default:
      assert(0);
      break;
   }

   ioAppend = parent->getCheckpointReadFlag();

   this->io_timer     = new Timer(getName(), "conn", "io     ");
   this->update_timer = new Timer(getName(), "conn", "update ");

   return status;
}

int PoolingConn::communicateInitInfo() {
   int status = HyPerConn::communicateInitInfo();

   //Check pre/post connections here
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const PVLayerLoc * postLoc = post->getLayerLoc();
   
   if(preLoc->nf != postLoc->nf){
      std::cout << "Pooling Layer " << name << " error:  preLayer " << pre->getName() << " nf of " << preLoc->nf << " does not match postLayer " << post->getName() << " nf of " << preLoc->nf << ". Features must match\n";
      exit(-1);
   }

   float preToPostScaleX = (float)preLoc->nx/postLoc->nx;
   float preToPostScaleY = (float)preLoc->ny/postLoc->ny;
   if(preToPostScaleX < 1 || preToPostScaleY < 1){
      std::cout << "Pooling Layer " << name << " error:  preLayer to postLayer must be a many to one or one to one conection\n";
      exit(-1);
   }

   //need postToPreBuffer
   setNeedPost(true);
   needAllocPostWeights = false;
   return status;
}

int PoolingConn::finalizeUpdate(double time, double dt) {
   return PV_SUCCESS;
}

int PoolingConn::allocateDataStructures(){
   int status = HyPerConn::allocateDataStructures();
   assert(status == PV_SUCCESS);
   return PV_SUCCESS;
}

int PoolingConn::setInitialValues() {
   //Doing nothing
   return PV_SUCCESS;
}

int PoolingConn::constructWeights(){
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();
   int status = PV_SUCCESS;

   assert(!parent->parameters()->presentAndNotBeenRead(name, "shrinkPatches"));
   // createArbors() uses the value of shrinkPatches.  It should have already been read in ioParamsFillGroup.
   //allocate the arbor arrays:
   createArbors();

   setPatchStrides();

   for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
      PVPatch *** wPatches = get_wPatches();
      status = createWeights(wPatches, arborId);
      assert(wPatches[arborId] != NULL);
      if (shrinkPatches_flag || arborId == 0){
         status |= adjustAxonalArbors(arborId);
      }
   }  // arborId

   //call to initializeWeights moved to BaseConnection::initializeState()
   status |= initPlasticityPatches();
   assert(status == 0);
   if (shrinkPatches_flag) {
      for (int arborId=0;arborId<numAxonalArborLists;arborId++) {
         shrinkPatches(arborId);
      }
   }

   return status;

}

int PoolingConn::checkpointRead(const char * cpDir, double * timeptr) {
   return PV_SUCCESS;
}

int PoolingConn::checkpointWrite(const char * cpDir) {
   return PV_SUCCESS;
}

float PoolingConn::minWeight(int arborId){
   if(getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
     return 1.0;
   }
   else if(getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING){
     int relative_XScale = (int) pow(2, pre->getXScale() - post->getXScale());
     int relative_YScale = (int) pow(2, pre->getYScale() - post->getYScale());
     return (1.0/(nxp*nyp*relative_XScale*relative_YScale));
   }
   else {
	   assert(0); // only possibilities are ACCUMULATE_MAXPOOLING and ACCUMULATE_SUMPOOLING
	   return 0.0; // gets rid of a compile warning
   }
}

float PoolingConn::maxWeight(int arborId){
   if(getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
     return 1.0;
   }
   else if(getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING){
     int relative_XScale = (int) pow(2, pre->getXScale() - post->getXScale());
     int relative_YScale = (int) pow(2, pre->getYScale() - post->getYScale());
     return (1.0/(nxp*nyp*relative_XScale*relative_YScale));
   }
   else {
	   assert(0); // only possibilities are ACCUMULATE_MAXPOOLING and ACCUMULATE_SUMPOOLING
	   return 0.0; // gets rid of a compile warning
   }
}

void PoolingConn::deliverOnePreNeuronActivity(int patchIndex, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr) {
   PVPatch * weights = getWeights(patchIndex, arbor);
   const int nk = weights->nx * fPatchSize();
   const int ny = weights->ny;
   const int sy  = getPostNonextStrides()->sy;       // stride in layer
   const int syw = yPatchStride();                   // stride in patch
   pvwdata_t * weightDataStart = NULL; 
   pvgsyndata_t * postPatchStart = postBufferStart + getGSynPatchStart(patchIndex, arbor);
   int offset = 0;
   int sf = 1;
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const int kfPre = featureIndex(patchIndex, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
   offset = kfPre;
   sf = fPatchSize();
   pvwdata_t w = 1.0;
   if(getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING){
     float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
     float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
     w = 1.0/(nxp*nyp*relative_XScale*relative_YScale);
   }
   for (int y = 0; y < ny; y++) {
     (accumulateFunctionPointer)(nk, postPatchStart + y*sy + offset, a, &w, auxPtr, sf);
   }
}

void PoolingConn::deliverOnePostNeuronActivity(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, uint4 * rngPtr){

   pvwdata_t * weightY = NULL; //No weights in pooling
   int sf = postConn->fPatchSize();
   int yPatchSize = postConn->yPatchSize();
   int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();

   const PVLayerLoc * postLoc = post->getLayerLoc();
   const int kfPost = featureIndex(kTargetExt, postLoc->nx + postLoc->halo.lt + postLoc->halo.rt, postLoc->ny + postLoc->halo.dn + postLoc->halo.up, postLoc->nf);
   int offset = kfPost;

   pvwdata_t w = 1.0;
   if(getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING){
     float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
     float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
     w = 1.0/(nxp*nyp*relative_XScale*relative_YScale);
   }
   for (int ky = 0; ky < yPatchSize; ky++){
      float * activityY = &(activityStartBuf[ky*inSy+offset]);
      (accumulateFunctionFromPostPointer)(numPerStride, gSynPatchPos, activityY, &w, dt_factor, rngPtr, sf);
   }
}

} // end namespace PV
