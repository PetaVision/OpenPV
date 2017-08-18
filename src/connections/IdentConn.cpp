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
   int status = BaseConnection::initialize(name, hc);
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

void IdentConn::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      initializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(name, "initializeFromCheckpointFlag");
   }
}

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

void IdentConn::ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      this->convertRateToSpikeCount = false;
      parent->parameters()->handleUnnecessaryParameter(
            name, "convertRateToSpikeCount", this->convertRateToSpikeCount);
   }
}

int IdentConn::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseConnection::communicateInitInfo(message);
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
   return status;
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
