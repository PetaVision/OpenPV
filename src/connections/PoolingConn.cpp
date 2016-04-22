/*
 * PoolingConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "PoolingConn.hpp"
#include <cstring>
#include <cmath>

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
   if(thread_gateIdxBuffer){
      for(int ti = 0; ti < parent->getNumThreads(); ti++){
         free(thread_gateIdxBuffer[ti]);
         thread_gateIdxBuffer[ti] = NULL;
      }
      free(thread_gateIdxBuffer);
      thread_gateIdxBuffer = NULL;
   }
   if(postIndexLayerName){
      free(postIndexLayerName);
   }
}

int PoolingConn::initialize_base() {
   //gateIdxBuffer = NULL;
   thread_gateIdxBuffer = NULL;
   needPostIndexLayer = false;
   postIndexLayerName = NULL;
   postIndexLayer = NULL;

   return PV_SUCCESS;
}

int PoolingConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_needPostIndexLayer(ioFlag);
   ioParam_postIndexLayerName(ioFlag);

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

void PoolingConn::ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag){
   parent->ioParamValue(ioFlag, name, "needPostIndexLayer", &needPostIndexLayer, needPostIndexLayer);
}

void PoolingConn::ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag) {
   if(needPostIndexLayer){
      parent->ioParamStringRequired(ioFlag, name, "postIndexLayerName", &postIndexLayerName);
   }
}

void PoolingConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "normalizeMethod", "none", false/*case_insensitive*/);
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
   case ACCUMULATE_AVGPOOLING:
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
      std::cout << "Pooling Layer " << name << " error:  preLayer " << pre->getName() << " nf of " << preLoc->nf << " does not match postLayer " << post->getName() << " nf of " << postLoc->nf << ". Features must match\n";
      exit(-1);
   }

   float preToPostScaleX = (float)preLoc->nx/postLoc->nx;
   float preToPostScaleY = (float)preLoc->ny/postLoc->ny;
   if(preToPostScaleX < 1 || preToPostScaleY < 1){
      std::cout << "Pooling Layer " << name << " error:  preLayer to postLayer must be a many to one or one to one conection\n";
      exit(-1);
   }

   if(needPostIndexLayer){
      BaseLayer * basePostIndexLayer = parent->getLayerFromName(this->postIndexLayerName);
      if (basePostIndexLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: postIndexLayerName \"%s\" does not refer to any layer in the column.\n", this->getKeyword(), name, this->postIndexLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      postIndexLayer = dynamic_cast<PoolingIndexLayer*>(basePostIndexLayer);
      if (postIndexLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: postIndexLayerName \"%s\" is not a PoolingIndexLayer.\n", this->getKeyword(), name, this->postIndexLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      if(postIndexLayer->getDataType() != PV_INT){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: postIndexLayer \"%s\" must have data type of int. Specify parameter dataType in this layer to be \"int\".\n", this->getKeyword(), name, this->postIndexLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
         
      }

      const PVLayerLoc * idxLoc = postIndexLayer->getLayerLoc();
      //postIndexLayer must be the same size as the post layer
      //(margins doesnt matter)
      if(idxLoc->nxGlobal != postLoc->nxGlobal || idxLoc->nyGlobal != postLoc->nyGlobal || idxLoc->nf != postLoc->nf){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: postIndexLayer \"%s\" must have the same dimensions as the post pooling layer \"%s\".", this->getKeyword(), name, this->postIndexLayerName, this->postLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      //TODO this is currently a hack, need to properly implement data types.
      assert(sizeof(int) == sizeof(float));
   }

   if(getUpdateGSynFromPostPerspective()){
      setNeedPost(true);
      needAllocPostWeights = false;
   }

   //if(needPostIndexLayer){
   //   //Synchronize margines of this post and the postIndexLayer, and vice versa
   //   post->synchronizeMarginWidth(postIndexLayer);
   //   postIndexLayer->synchronizeMarginWidth(post);
   //}


   return status;
}

int PoolingConn::finalizeUpdate(double time, double dt) {
   return PV_SUCCESS;
}

void PoolingConn::clearGateIdxBuffer(){
   if(needPostIndexLayer){
      //Reset postIndexLayer's gsyn
      resetGSynBuffers_PoolingIndexLayer(parent->getNBatch(), postIndexLayer->getNumNeurons(), postIndexLayer->getNumChannels(), postIndexLayer->getChannel(CHANNEL_EXC)); // resetGSynBuffers();
   }
}

int PoolingConn::allocateDataStructures(){
   int status = HyPerConn::allocateDataStructures();
   if (status == PV_POSTPONE) { return status; }
   assert(status == PV_SUCCESS);

   if(needPostIndexLayer){
      //Allocate temp buffers if needed, 1 for each thread
      if(parent->getNumThreads() > 1){
         thread_gateIdxBuffer= (int**) malloc(sizeof(int*) * parent->getNumThreads());
         //thread_gateIdxBuffer= (float**) malloc(sizeof(float*) * parent->getNumThreads());
         assert(thread_gateIdxBuffer);
         //Assign thread_gSyn to different points of tempMem
         for(int i = 0; i < parent->getNumThreads(); i++){
            int* thread_buffer = (int*) malloc(sizeof(int) * post->getNumNeurons());
            //float* thread_buffer = (float*) malloc(sizeof(float) * post->getNumNeurons());
            if(!thread_buffer){
               fprintf(stderr, "HyPerLayer \"%s\" error: rank %d unable to allocate %zu memory for thread_gateIdxBuffer: %s\n", name, parent->columnId(), sizeof(int) * post->getNumNeurons(), strerror(errno));
               exit(EXIT_FAILURE);
            }
            thread_gateIdxBuffer[i] = thread_buffer;
         }
      }

      if(thread_gateIdxBuffer){
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int i = 0; i < parent->getNumThreads() * post->getNumNeurons(); i++){
            int ti = i/post->getNumNeurons();
            int ni = i % post->getNumNeurons();
            thread_gateIdxBuffer[ti][ni] = -1;
         }
      }

      //gateIdxBuffer = (int*)malloc(post->getNumNeurons() * sizeof(int));
      //assert(gateIdxBuffer);

      clearGateIdxBuffer();
   }
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
     return 1;
   }
   else if(getPvpatchAccumulateType() == ACCUMULATE_AVGPOOLING){
     int relative_XScale = (int) pow(2, pre->getXScale() - post->getXScale());
     int relative_YScale = (int) pow(2, pre->getYScale() - post->getYScale());
     return (1.0/(nxp*nyp*relative_XScale*relative_YScale));
   }
   else {
       assert(0); // only possibilities are ACCUMULATE_MAXPOOLING, ACCUMULATE_SUMPOOLING, ACCUMULATe_AVGPOOLING
       return 0.0; // gets rid of a compile warning
    }
}

float PoolingConn::maxWeight(int arborId){
   if(getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
     return 1.0;
   }
   else if(getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING){
     return 1;
   }
   else if(getPvpatchAccumulateType() == ACCUMULATE_AVGPOOLING){
     int relative_XScale = (int) pow(2, pre->getXScale() - post->getXScale());
     int relative_YScale = (int) pow(2, pre->getYScale() - post->getYScale());
     return (1.0/(nxp*nyp*relative_XScale*relative_YScale));
   }
   else {
       assert(0); // only possibilities are ACCUMULATE_MAXPOOLING and ACCUMULATE_SUMPOOLING
       return 0.0; // gets rid of a compile warning
    }
}

int PoolingConn::deliverPresynapticPerspective(PVLayerCube const * activity, int arborID) {

   //Check if we need to update based on connection's channel
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));

   float dt_factor;
   if (getPvpatchAccumulateType()==ACCUMULATE_STOCHASTIC) {
      dt_factor = getParent()->getDeltaTime();
   }
   else {
      dt_factor = getConvertToRateDeltaTimeFactor();
   }

   const PVLayerLoc * preLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = postSynapticLayer()->getLayerLoc();

   assert(arborID >= 0);
   const int numExtended = activity->numItems;

   float resetVal = 0;
   if(getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
      resetVal = -INFINITY;
      float* gSyn = post->getChannel(getChannel());
      //gSyn is res
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int i = 0; i < post->getNumNeuronsAllBatches(); i++){
         gSyn[i] = resetVal;
      }
      
   }


   clearGateIdxBuffer();

   for(int b = 0; b < parent->getNBatch(); b++){
      pvdata_t * activityBatch = activity->data + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      pvdata_t * gSynPatchHeadBatch = post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
      int* gatePatchHeadBatch = NULL;
      if(needPostIndexLayer){
         gatePatchHeadBatch = postIndexLayer->getChannel(CHANNEL_EXC) + b * postIndexLayer->getNumNeurons();
      }

      unsigned int * activeIndicesBatch = NULL;
      if(activity->isSparse){
         activeIndicesBatch = activity->activeIndices + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      }
      int numLoop;
      if(activity->isSparse){
         numLoop = activity->numActive[b];
      }
      else{
         numLoop = numExtended;
      }

      if(thread_gateIdxBuffer){
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int i = 0; i < parent->getNumThreads() * post->getNumNeurons(); i++){
            int ti = i/post->getNumNeurons();
            int ni = i % post->getNumNeurons();
            thread_gateIdxBuffer[ti][ni] = -1;
         }
      }

#ifdef PV_USE_OPENMP_THREADS
      //Clear all gsyn buffers
      if(thread_gSyn){
         int numNeurons = post->getNumNeurons();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int i = 0; i < parent->getNumThreads() * numNeurons; i++){
            int ti = i/numNeurons;
            int ni = i % numNeurons;
            thread_gSyn[ti][ni] = resetVal;
         }
      }
#endif // PV_USE_OPENMP_THREADS
      
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
         int kPreExt;
         if(activity->isSparse){
            kPreExt = activeIndicesBatch[loopIndex];
         }
         else{
            kPreExt = loopIndex;
         }

         float a = activityBatch[kPreExt] * dt_factor;
         //if (a == 0.0f) continue;

         //If we're using thread_gSyn, set this here
         pvdata_t * gSynPatchHead;
         //float * gatePatchHead = NULL;
         int * gatePatchHead = NULL;
#ifdef PV_USE_OPENMP_THREADS
         if(thread_gSyn){
            int ti = omp_get_thread_num();
            gSynPatchHead = thread_gSyn[ti];
         }
         else{
            gSynPatchHead = gSynPatchHeadBatch;
         }

         if(needPostIndexLayer){
            if(thread_gateIdxBuffer){
               int ti = omp_get_thread_num();
               gatePatchHead = thread_gateIdxBuffer[ti];
            }
            else{
               gatePatchHead = gatePatchHeadBatch;
            }
         }
#else // PV_USE_OPENMP_THREADS
         gSynPatchHead = gSynPatchHeadBatch;
         if(needPostIndexLayer){
            gatePatchHead = gatePatchHeadBatch;
         }
#endif // PV_USE_OPENMP_THREADS
         //deliverOnePreNeuronActivity(kPreExt, arborID, a, gSynPatchHead, gatePatchHead);
         
         PVPatch * weights = getWeights(kPreExt, arborID);
         const int nk = weights->nx * fPatchSize();
         const int ny = weights->ny;
         const int sy  = getPostNonextStrides()->sy;       // stride in layer
         pvwdata_t * weightDataStart = NULL; 
         pvgsyndata_t * postPatchStart = gSynPatchHead + getGSynPatchStart(kPreExt, arborID);
         int* postGatePatchStart = gatePatchHead + getGSynPatchStart(kPreExt, arborID);
         //float* postGatePatchStart = gatePatchHead + getGSynPatchStart(kPreExt, arborID);

         const int kxPreExt = kxPos(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
         const int kyPreExt = kyPos(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);
         const int kfPre = featureIndex(kPreExt, preLoc->nx + preLoc->halo.lt + preLoc->halo.rt, preLoc->ny + preLoc->halo.dn + preLoc->halo.up, preLoc->nf);

         const int kxPreGlobalExt = kxPreExt + preLoc->kx0;
         const int kyPreGlobalExt = kyPreExt + preLoc->ky0;

         const int kPreGlobalExt = kIndex(kxPreGlobalExt, kyPreGlobalExt, kfPre, preLoc->nxGlobal + preLoc->halo.lt + preLoc->halo.rt, preLoc->nyGlobal + preLoc->halo.up + preLoc->halo.dn, preLoc->nf);

         int offset = kfPre;
         int sf = fPatchSize();
         pvwdata_t w = 1.0;
         if(getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING){
           w = 1.0;
         }
         else if(getPvpatchAccumulateType() == ACCUMULATE_AVGPOOLING){
           float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
           float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
           w = 1.0/(nxp*nyp*relative_XScale*relative_YScale);
         }
         void* auxPtr = NULL;
         for (int y = 0; y < ny; y++) {
            if(needPostIndexLayer){
               auxPtr = (postGatePatchStart+ y*sy + offset);
            }
            (accumulateFunctionPointer)(kPreGlobalExt, nk, postPatchStart + y*sy + offset, a, &w, auxPtr, sf);
         }
      }
#ifdef PV_USE_OPENMP_THREADS
      //Accumulate back into gSyn // Should this be done in HyPerLayer where it can be done once, as opposed to once per connection?
      if(thread_gSyn){
         pvdata_t * gSynPatchHead = gSynPatchHeadBatch;
         //float* gateIdxBuffer = postIndexLayer->getChannel(CHANNEL_EXC);
         int * gateIdxBuffer = NULL;
         if(needPostIndexLayer && thread_gateIdxBuffer){
            gateIdxBuffer = gatePatchHeadBatch;
         }
         int numNeurons = post->getNumNeurons();
         //Looping over neurons first to be thread safe
#pragma omp parallel for
         for(int ni = 0; ni < numNeurons; ni++){
            //Different for maxpooling
            if(getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
               for(int ti = 0; ti < parent->getNumThreads(); ti++){
                  if(gSynPatchHead[ni] < thread_gSyn[ti][ni]){
                     gSynPatchHead[ni] = thread_gSyn[ti][ni];
                     if(needPostIndexLayer && thread_gateIdxBuffer){
                        gateIdxBuffer[ni] = thread_gateIdxBuffer[ti][ni]; 
                        assert(gateIdxBuffer >= 0);
                     }
                  }
               }
            }
            else{
               for(int ti = 0; ti < parent->getNumThreads(); ti++){
                  gSynPatchHead[ni] += thread_gSyn[ti][ni];
               }
            }
         }
      }
#endif
   }
   if(activity->isSparse){
      pvdata_t * gSyn = post->getChannel(getChannel());
      for (int k=0; k<post->getNumNeuronsAllBatches(); k++) {
         if (gSyn[k]==-INFINITY) {
            gSyn[k] = 0.0f;
         }
      }
   }
   return PV_SUCCESS;
}

int PoolingConn::deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID) {
   //Check channel number for noupdate
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));

   assert(arborID >= 0);
   //Get number of neurons restricted target
   const int numPostRestricted = post->getNumNeurons();

   float dt_factor = getConvertToRateDeltaTimeFactor();

   const PVLayerLoc * sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * targetLoc = post->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   const PVHalo * sourceHalo = &sourceLoc->halo;
   const PVHalo * targetHalo = &targetLoc->halo;

   //get source layer's extended y stride
   int sy  = (sourceNx+sourceHalo->lt+sourceHalo->rt)*sourceNf;

   //The start of the gsyn buffer
   pvdata_t * gSynPatchHead = post->getChannel(this->getChannel());

   clearGateIdxBuffer();
   int* gatePatchHead = NULL;
   if(needPostIndexLayer){
      gatePatchHead = postIndexLayer->getChannel(CHANNEL_EXC);
   }


   long * startSourceExtBuf = getPostToPreActivity();
   if(!startSourceExtBuf){
      std::cout << "HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?\n";
      exit(EXIT_FAILURE);
   }

   float resetVal = 0;
   if(getPvpatchAccumulateType() == ACCUMULATE_MAXPOOLING){
      resetVal = -INFINITY;
   }


   for(int b = 0; b < parent->getNBatch(); b++){
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kTargetRes = 0; kTargetRes < numPostRestricted; kTargetRes++){
         pvdata_t * activityBatch = activity->data + b * (sourceNx + sourceHalo->rt + sourceHalo->lt) * (sourceNy + sourceHalo->up + sourceHalo->dn) * sourceNf;
         pvdata_t * gSynPatchHeadBatch = gSynPatchHead + b * targetNx * targetNy * targetNf;
         
         //Change restricted to extended post neuron
         int kTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);

         //Read from buffer
         long startSourceExt = startSourceExtBuf[kTargetRes];

         //Calculate target's start of gsyn
         pvdata_t * gSynPatchPos = gSynPatchHeadBatch + kTargetRes;
         //Initialize patch as a huge negative number
         *gSynPatchPos = resetVal;

         int* gatePatchPos = NULL;
         if(needPostIndexLayer){
            gatePatchPos = gatePatchHead + b * postIndexLayer->getNumNeurons() + kTargetRes;
            //Initialize gatePatchPos as a negative number
            *gatePatchPos = -1;
         }

         float* activityStartBuf = &(activityBatch[startSourceExt]); 

         pvwdata_t * weightY = NULL; //No weights in pooling
         int sf = postConn->fPatchSize();
         int yPatchSize = postConn->yPatchSize();
         int numPerStride = postConn->xPatchSize() * postConn->fPatchSize();

         const PVLayerLoc * postLoc = post->getLayerLoc();
         const int kfPost = featureIndex(kTargetExt, postLoc->nx + postLoc->halo.lt + postLoc->halo.rt, postLoc->ny + postLoc->halo.dn + postLoc->halo.up, postLoc->nf);
         int offset = kfPost;

         pvwdata_t w = 1.0;
         if(getPvpatchAccumulateType() == ACCUMULATE_SUMPOOLING){
           w = 1.0;
         }
         else if(getPvpatchAccumulateType() == ACCUMULATE_AVGPOOLING){
           float relative_XScale = pow(2, (post->getXScale() - pre->getXScale()));
           float relative_YScale = pow(2, (post->getYScale() - pre->getYScale()));
           w = 1.0/(nxp*nyp*relative_XScale*relative_YScale);
         }

         for (int ky = 0; ky < yPatchSize; ky++){
            int kPreExt = startSourceExt + ky*sy+offset;
            const int kxPreExt = kxPos(kPreExt, sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt, sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up, sourceLoc->nf);
            const int kyPreExt = kyPos(kPreExt, sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt, sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up, sourceLoc->nf);
            const int kfPre = featureIndex(kPreExt, sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt, sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up, sourceLoc->nf);
            const int kxPreGlobalExt = kxPreExt + sourceLoc->kx0;
            const int kyPreGlobalExt = kyPreExt + sourceLoc->ky0;
            const int kPreGlobalExt = kIndex(kxPreGlobalExt, kyPreGlobalExt, kfPre, sourceLoc->nxGlobal + sourceLoc->halo.lt + sourceLoc->halo.rt, sourceLoc->nyGlobal + sourceLoc->halo.up + sourceLoc->halo.dn, sourceLoc->nf);

            float * activityY = &(activityStartBuf[ky*sy+offset]);

            (accumulateFunctionFromPostPointer)(kPreGlobalExt, numPerStride, gSynPatchPos, activityY, &w, dt_factor, gatePatchPos, sf);
         }
      }
   }
   return PV_SUCCESS;
}

BaseObject * createPoolingConn(char const * name, HyPerCol * hc) {
   return hc ? new PoolingConn(name, hc) : NULL;
}

} // end namespace PV
