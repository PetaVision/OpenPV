/*
 * IdentConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "IdentConn.hpp"

namespace PV {

IdentConn::IdentConn() { initialize_base(); }

IdentConn::IdentConn(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int IdentConn::initialize_base() {
   // no IdentConn-specific data members to initialize
   return PV_SUCCESS;
} // end of IdentConn::initialize_base()

int IdentConn::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

#ifdef PV_USE_CUDA
void IdentConn::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   // Never receive from gpu
   receiveGpu = false;
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "receiveGpu", false /*correctValue*/);
   }
}
#endif // PV_USE_CUDA

// Note this is one of the subclasses of the former kernelconn where it doesn't make sense to allow
// sharedWeights to be false
void IdentConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(
            name, "sharedWeights", true /*correctValue*/);
   }
}

void IdentConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
}

void IdentConn::ioParam_weightInitType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryStringParameter(name, "weightInitType", NULL);
      weightInitializer = nullptr;
   }
}

void IdentConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) { normalizer = NULL; }

void IdentConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      numAxonalArborLists = 1;
      parent->parameters()->handleUnnecessaryParameter(
            name, "numAxonalArbors", numAxonalArborLists);
   }
}

void IdentConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      plasticityFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
   }
}

void IdentConn::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   // IdentConn does not use the pvpatchAccumulateType parameter
   if (ioFlag == PARAMS_IO_READ) {
      pvpatchAccumulateTypeString = strdup("convolve");
      pvpatchAccumulateType       = CONVOLVE;
      parent->parameters()->handleUnnecessaryStringParameter(
            name, "pvpatchAccumulateType", "convolve", true /*case insensitive*/);
   }
}
void IdentConn::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      if (parent->parameters()->present(name, "writeStep")) {
         parent->parameters()->ioParamValue(
               ioFlag, name, "writeStep", &writeStep, -1.0 /*default*/, false /*warnIfAbsent*/);
         if (writeStep >= 0) {
            if (parent->columnId() == 0) {
               ErrorLog().printf(
                     "%s does not use writeStep, but the parameters file sets it to %f.\n",
                     getDescription_c(),
                     writeStep);
            }
            MPI_Barrier(parent->getCommunicator()->communicator());
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
      parent->parameters()->handleUnnecessaryParameter(
            name, "convertRateToSpikeCount", this->convertRateToSpikeCount);
   }
}

void IdentConn::ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeCompressedWeights = true;
      parent->parameters()->handleUnnecessaryParameter(
            name, "writeCompressedWeights", writeCompressedWeights);
   }
}
void IdentConn::ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeCompressedCheckpoints = true;
      parent->parameters()->handleUnnecessaryParameter(
            name, "writeCompressedCheckpoints", writeCompressedCheckpoints);
   }
}

void IdentConn::ioParam_selfFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      selfFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "selfFlag", selfFlag);
   }
}

void IdentConn::ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) {
   assert(plasticityFlag == false);
   // readCombine_dW_with_W_flag only used if when plasticityFlag is true, which it never is for
   // IdentConn
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(
            name, "combine_dW_with_W_flag", combine_dW_with_W_flag);
   }
   return;
}

void IdentConn::ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      weightUpdatePeriod = 1.0f;
      parent->parameters()->handleUnnecessaryParameter(
            name, "weightUpdatePeriod", weightUpdatePeriod);
   }
}

void IdentConn::ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initialWeightUpdateTime = 0.0f;
      parent->parameters()->handleUnnecessaryParameter(
            name, "initialWeightUpdateTime", initialWeightUpdateTime);
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

void IdentConn::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      updateGSynFromPostPerspective = false;
      parent->parameters()->handleUnnecessaryParameter(
            name, "updateGSynFromPostPerspective", updateGSynFromPostPerspective);
   }
}

int IdentConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = HyPerConn::communicateInitInfo(message);
   assert(pre && post);
   const PVLayerLoc *preLoc  = pre->getLayerLoc();
   const PVLayerLoc *postLoc = post->getLayerLoc();
   if (preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny || preLoc->nf != postLoc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "IdentConn \"%s\" Error: %s and %s do not have the same dimensions.\n Dims: "
               "%dx%dx%d vs. %dx%dx%d\n",
               name,
               preLayerName,
               postLayerName,
               preLoc->nx,
               preLoc->ny,
               preLoc->nf,
               postLoc->nx,
               postLoc->ny,
               postLoc->nf);
      }
      exit(EXIT_FAILURE);
   }
   parent->parameters()->handleUnnecessaryParameter(
         name, "nfp", nfp); // nfp is set during call to HyPerConn::communicateInitInfo, so don't
   // check for unnecessary int parameter until after that.
   return status;
}

void IdentConn::handleDefaultSelfFlag() { assert(selfFlag == false); }

int IdentConn::registerData(Checkpointer *checkpointer) {
   registerTimers(checkpointer);
   return PV_SUCCESS;
}

int IdentConn::deliver() {
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   float *postChannel = post->getChannel(getChannel());
   pvAssert(postChannel);

   pvAssert(numberOfAxonalArborLists() == 1);

   int const delay                   = getDelay(0);
   HyPerLayer *pre                   = preSynapticLayer();
   PVLayerCube const preActivityCube = pre->getPublisher()->createCube(delay);

   HyPerLayer *post = postSynapticLayer();
   pvAssert(pre->getNumNeurons() == post->getNumNeurons());

   PVLayerLoc const *preLoc  = &preActivityCube.loc;
   PVLayerLoc const *postLoc = post->getLayerLoc();
   pvAssert(preLoc->nx == postLoc->nx and preLoc->ny == postLoc->ny and preLoc->nf == postLoc->nf);
   int const nx       = preLoc->nx;
   int const ny       = preLoc->ny;
   int const nf       = preLoc->nf;
   int nxPreExtended  = nx + preLoc->halo.lt + preLoc->halo.rt;
   int nyPreExtended  = ny + preLoc->halo.dn + preLoc->halo.up;
   int numPreExtended = nxPreExtended * nyPreExtended * nf;
   pvAssert(numPreExtended == pre->getNumExtended());

   for (int b = 0; b < parent->getNBatch(); b++) {
      float const *preActivityBuffer = preActivityCube.data + b * numPreExtended;
      float *postGSynBuffer          = postChannel + b * post->getNumNeurons();
      if (preActivityCube.isSparse) {
         SparseList<float>::Entry const *activeIndices =
               (SparseList<float>::Entry *)preActivityCube.activeIndices + b * numPreExtended;
         int numActive = preActivityCube.numActive[b];
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int loopIndex = 0; loopIndex < numActive; loopIndex++) {
            int kPre = activeIndices[loopIndex].index;
            int kx   = kxPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc->halo.lt;
            int ky   = kyPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc->halo.up;
            if (kx < 0 or kx >= nx or ky < 0 or kx >= ny) {
               continue;
            }
            int kf    = featureIndex(kPre, nxPreExtended, nyPreExtended, nf) - preLoc->halo.up;
            int kPost = kIndex(kx, ky, kf, nx, ny, nf);
            pvAssert(kPost >= 0 and kPost < post->getNumNeurons());
            float a = activeIndices[loopIndex].value;
            postGSynBuffer[kPost] += a;
         }
      }
      else {
         int const nk = postLoc->nx * postLoc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int y = 0; y < ny; y++) {
            int preLineIndex =
                  kIndex(preLoc->halo.lt, y + preLoc->halo.up, 0, nxPreExtended, nyPreExtended, nf);

            float const *preActivityLine = &preActivityBuffer[preLineIndex];
            int postLineIndex            = kIndex(0, y, 0, postLoc->nx, ny, postLoc->nf);
            float *postGSynLine          = &postGSynBuffer[postLineIndex];
            for (int k = 0; k < nk; k++) {
               postGSynLine[k] += preActivityLine[k];
            }
         }
      }
   }
   return PV_SUCCESS;
}

} // end of namespace PV block
