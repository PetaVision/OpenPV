/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"
#include "weightinit/InitIdentWeights.hpp"
#include "utils/PVLog.hpp"

namespace PV {

IdentConn::IdentConn() {
    initialize_base();
}

IdentConn::IdentConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int IdentConn::initialize_base() {
   // no IdentConn-specific data members to initialize
   return PV_SUCCESS;
}  // end of IdentConn::initialize_base()

int IdentConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc, NULL, NULL);
   return status;
}

int IdentConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);

   // April 15, 2016: Scale moved from IdentConn to RescaleConn.
   if (!strcmp(getKeyword(), "IdentConn") && parent->parameters()->present(name, "scale")) {
      logError("IdentConn \"%s\" error: IdentConn does not take a scale parameter.  Use RescaleConn instead.\n", name);
   }

   return status;
}

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
void IdentConn::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   //Never receive from gpu
   receiveGpu = false;
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "receiveGpu", false/*correctValue*/);
   }
}
#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)


// Note this is one of the subclasses of the former kernelconn where it doesn't make sense to allow sharedWeights to be false
void IdentConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void IdentConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights");
   }
}

void IdentConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      weightInitializer = new InitIdentWeights(name, parent);
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
   }
}

void IdentConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   normalizer = NULL;
}

void IdentConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      numAxonalArborLists = 1;
      parent->parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", numAxonalArborLists);
   }
}

void IdentConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
   }
}

void IdentConn::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      pvpatchAccumulateTypeString = strdup("convolve");
      pvpatchAccumulateType = ACCUMULATE_CONVOLVE;
      parent->parameters()->handleUnnecessaryStringParameter(name, "pvpatchAccumulateType", "convolve", true/*case insensitive*/);
   }
}
void IdentConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      if (parent->parameters()->present(name, "writeStep")) {
         parent->ioParamValue(ioFlag, name, "writeStep", &writeStep, -1.0/*default*/, false/*warnIfAbsent*/);
         if (writeStep>=0) {
            if (parent->columnId()==0) {
               fprintf(stderr, "Error: %s \"%s\" does not use writeStep, but the parameters file sets it to %f.\n", getKeyword(), getName(), writeStep);
            }
            MPI_Barrier(parent->icCommunicator()->communicator());
            exit(EXIT_FAILURE);
         }
      }
      else {
         writeStep = -1.0;
         parent->parameters()->handleUnnecessaryParameter(name, "writeStep");
      }
   }
}

void IdentConn::ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      this->convertRateToSpikeCount = false;
      parent->parameters()->handleUnnecessaryParameter(name, "convertRateToSpikeCount", this->convertRateToSpikeCount);
   }
}

void IdentConn::ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeCompressedWeights = true;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedWeights", writeCompressedWeights);
   }
}
void IdentConn::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeCompressedCheckpoints = true;
      parent->parameters()->handleUnnecessaryParameter(name, "writeCompressedCheckpoints", writeCompressedCheckpoints);
   }
}

void IdentConn::ioParam_selfFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      selfFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "selfFlag", selfFlag);
   }
}

void IdentConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   assert(plasticityFlag==false);
   // readCombine_dW_with_W_flag only used if when plasticityFlag is true, which it never is for IdentConn
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "combine_dW_with_W_flag", combine_dW_with_W_flag);
   }
   return;
}

void IdentConn::ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      keepKernelsSynchronized_flag = true;
      parent->parameters()->handleUnnecessaryParameter(name, "keepKernelsSynchronized", keepKernelsSynchronized_flag);
   }
}

void IdentConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = 1.0f;
      parent->parameters()->handleUnnecessaryParameter(name, "weightUpdatePeriod", weightUpdatePeriod);
   }
}

void IdentConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = 0.0f;
      parent->parameters()->handleUnnecessaryParameter(name, "initialWeightUpdateTime", initialWeightUpdateTime);
      weightUpdateTime = initialWeightUpdateTime;
   }
}

void IdentConn::ioParam_nxp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      nxp = 1;
      parent->parameters()->handleUnnecessaryParameter(name, "nxp", 1);
   }
}

void IdentConn::ioParam_nyp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      nyp = 1;
      parent->parameters()->handleUnnecessaryParameter(name, "nyp", 1);
   }
}

void IdentConn::ioParam_nfp(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "nfp", -1);
   }
}

void IdentConn::ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      shrinkPatches_flag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches", shrinkPatches_flag);
   }
}

void IdentConn::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag){
   if (ioFlag == PARAMS_IO_READ) {
      updateGSynFromPostPerspective = false;
      parent->parameters()->handleUnnecessaryParameter(name, "updateGSynFromPostPerspective", updateGSynFromPostPerspective);
   }
}

int IdentConn::setWeightInitializer() {
   weightInitializer = (InitWeights *) new InitIdentWeights(name, parent);
   if( weightInitializer == NULL ) {
      fprintf(stderr, "IdentConn \"%s\": Rank %d process unable to create InitIdentWeights object.  Exiting.\n", name, parent->icCommunicator()->commRank());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int IdentConn::communicateInitInfo() {
   int status = HyPerConn::communicateInitInfo();
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
   parent->parameters()->handleUnnecessaryParameter(name, "nfp", nfp); // nfp is set during call to HyPerConn::communicateInitInfo, so don't check for unnecessary int parameter until after that.
   return status;
}

void IdentConn::handleDefaultSelfFlag() {
   assert(selfFlag==false);
}

//This deliver only differs from HyPerConn's deliver through if statements checking for recv GPU and recv from post. Since these flags are hard coded in IdentConn, calling HyPerConn's deliver should be okay. This is to make deliver a non-virtual method, as HyPerConn's methods sets flags for GPU updates of GSyn, a source of errors when a subclass overwrites deliver.
//int IdentConn::deliver() {
//   int status = PV_SUCCESS;
//
//   //Check if updating from post perspective
//   HyPerLayer * pre = preSynapticLayer();
//   PVLayerCube cube;
//   memcpy(&cube.loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
//   cube.numItems = pre->getNumExtended();
//   cube.size = sizeof(PVLayerCube);
//
//   DataStore * store = parent->icCommunicator()->publisherStore(pre->getLayerId());
//   assert(numberOfAxonalArborLists()==1);
//   int arbor = 0;
//
//   int delay = getDelay(arbor);
//   cube.data = (pvdata_t *) store->buffer(LOCAL, delay);
//   assert(getUpdateGSynFromPostPerspective()==false);
//#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
//   assert(getReceiveGpu()==false);
//#endif // defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
//
//   cube.isSparse = store->isSparse();
//   if(cube.isSparse){
//      cube.numActive = store->numActiveBuffer(LOCAL, delay);
//      cube.activeIndices = store->activeIndicesBuffer(LOCAL, delay);
//   }
//
//   status = this->deliverPresynapticPerspective(&cube, arbor);
//   //IdentConns never deliver from gpu, so always set this flag to true
//   post->setUpdatedDeviceGSynFlag(true);
//
//   assert(status == PV_SUCCESS);
//   return PV_SUCCESS;
//}

int IdentConn::deliverPresynapticPerspective(PVLayerCube const * activity, int arborID) {

   //Check if we need to update based on connection's channel
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   assert(post->getChannel(getChannel()));

   float dt_factor = getConvertToRateDeltaTimeFactor();
   assert(dt_factor==1.0);

   const PVLayerLoc * preLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = postSynapticLayer()->getLayerLoc();

   assert(arborID==0); // IdentConn can only have one arbor
   const int numExtended = activity->numItems;

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(parent->icCommunicator()->communicator(), &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
   fflush(stdout);
#endif // DEBUG_OUTPUT

   for(int b = 0; b < parent->getNBatch(); b++){
      pvdata_t * activityBatch = activity->data + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      pvdata_t * gSynPatchHeadBatch = post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;

      if (activity->isSparse) {
         unsigned int * activeIndicesBatch = activity->activeIndices + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
         int numLoop = activity->numActive[b];
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
            int kPre = activeIndicesBatch[loopIndex];

            float a = activityBatch[kPre];
            // if (a == 0.0f) continue;
            PVPatch * weights = getWeights(kPre, arborID);
            if (weights->nx>0 && weights->ny>0) {
               int f = featureIndex(kPre, preLoc->nx, preLoc->ny, preLoc->nf); // Not taking halo into account, but for feature index, shouldn't matter.
               pvgsyndata_t * postPatchStart = gSynPatchHeadBatch + getGSynPatchStart(kPre, arborID) + f;
               *postPatchStart += a;
            }
         }
      }
      else {
         PVLayerLoc const * loc = &activity->loc;
         PVHalo const * halo = &loc->halo;
         // The code below is a replacement for the block marked obsolete below it.  Jan 5, 2016
         int lineSizeExt = (loc->nx+halo->lt+halo->rt)*loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int y=0; y<loc->ny; y++) {
            pvdata_t * lineStartPreActivity = &activityBatch[(y+halo->up)*lineSizeExt+halo->lt*loc->nf];
            int nk = loc->nx*loc->nf;
            pvdata_t * lineStartPostGSyn = &gSynPatchHeadBatch[y*nk];
            for (int k=0; k<nk; k++) {
               lineStartPostGSyn[k] += lineStartPreActivity[k];
            }
         }
#ifdef OBSOLETE // Marked obsolete Jan 5, 2016.  IdentConn is simple enough that we shouldn't need to call kIndexExtended inside the inner loop.
         int numRestricted = loc->nx * loc->ny * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kRestricted = 0; kRestricted < numRestricted; kRestricted++) {
            int kExtended = kIndexExtended(kRestricted, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
            float a = activityBatch[kExtended];
            // if (a == 0.0f) continue;
            gSynPatchHeadBatch[kRestricted] += a;
         }
#endif // OBSOLETE // Marked obsolete Jan 5, 2016
      }
   }
   return PV_SUCCESS;
}

BaseObject * createIdentConn(char const * name, HyPerCol * hc) {
   return hc ? new IdentConn(name, hc) : NULL;
}

}  // end of namespace PV block
