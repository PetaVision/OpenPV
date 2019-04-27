/*
 * MaskActivityBuffer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#include "MaskActivityBuffer.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/ActivityComponent.hpp"

namespace PV {

MaskActivityBuffer::MaskActivityBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

MaskActivityBuffer::MaskActivityBuffer() {}

MaskActivityBuffer::~MaskActivityBuffer() {
   free(mMaskLayerName);
   free(mFeatures);
   free(mMaskMethod);
}

int MaskActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_maskMethod(ioFlag);
   ioParam_maskLayerName(ioFlag);
   ioParam_featureIdxs(ioFlag);
   return status;
}

void MaskActivityBuffer::ioParam_maskMethod(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "maskMethod", &mMaskMethod);
   // Check valid methods
   if (strcmp(mMaskMethod, "layer") == 0) {
      mMaskMethodCode = LAYER;
   }
   else if (strcmp(mMaskMethod, "invertLayer") == 0) {
      mMaskMethodCode = INVERT_LAYER;
   }
   else if (strcmp(mMaskMethod, "maskFeatures") == 0) {
      mMaskMethodCode = FEATURES;
   }
   else if (strcmp(mMaskMethod, "noMaskFeatures") == 0) {
      mMaskMethodCode = INVERT_FEATURES;
   }
   if (mCommunicator->commRank() == 0) {
      FatalIf(
            mMaskMethodCode == UNDEFINED,
            "%s: \"%s\" is not a valid maskMethod. Options are \"layer\", \"invertLayer\", "
            "\"maskFeatures\", or \"noMaskFeatures\".\n",
            getDescription_c(),
            mMaskMethod);
   }
}

void MaskActivityBuffer::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "maskMethod"));
   if (mMaskMethodCode == LAYER or mMaskMethodCode == INVERT_LAYER) {
      parameters()->ioParamStringRequired(ioFlag, name, "maskLayerName", &mMaskLayerName);
   }
}

void MaskActivityBuffer::ioParam_featureIdxs(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "maskMethod"));
   if (mMaskMethodCode == FEATURES or mMaskMethodCode == INVERT_FEATURES) {
      parameters()->ioParamArray(ioFlag, name, "featureIdxs", &mFeatures, &mNumSpecifiedFeatures);
      if (mNumSpecifiedFeatures == 0) {
         if (mCommunicator->commRank() == 0) {
            ErrorLog().printf(
                  "%s: MaskLayer must specify at least one feature for maskMethod \"%s\".\n",
                  getDescription_c(),
                  mMaskMethod);
         }
         exit(EXIT_FAILURE);
      }
   }
}

Response::Status
MaskActivityBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ANNActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mMaskMethodCode == LAYER or mMaskMethodCode == INVERT_LAYER) {
      mMaskBuffer = message->mObjectTable->findObject<ActivityBuffer>(std::string(mMaskLayerName));
      FatalIf(
            mMaskBuffer == nullptr,
            "%s: No object with maskLayerName \"%s\" has an ActivityBuffer.\n",
            getDescription_c(),
            mMaskLayerName);
   }
   else {
      pvAssert(mMaskMethodCode == FEATURES or mMaskMethodCode == INVERT_FEATURES);
      // Check for in bounds featureIdxs
      assert(mFeatures);
      const PVLayerLoc *loc = getLayerLoc();
      for (int f = 0; f < mNumSpecifiedFeatures; f++) {
         if (mFeatures[f] < 0 || mFeatures[f] >= loc->nf) {
            Fatal() << "Specified feature " << mFeatures[f] << "out of bounds\n";
         }
      }
   }

   return Response::SUCCESS;
}

Response::Status MaskActivityBuffer::allocateDataStructures() {
   auto status = ANNActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (mMaskMethodCode == LAYER or mMaskMethodCode == INVERT_LAYER) {
      // MaskBuffer must have the same x and y dimensions as those of this buffer.
      // MaskLayer's nf must be either 1 or the same as this layer's nf.
      PVLayerLoc const *maskLoc = mMaskBuffer->getLayerLoc();
      PVLayerLoc const *loc     = getLayerLoc();
      pvAssert(maskLoc != nullptr && loc != nullptr);
      if (maskLoc->nxGlobal != loc->nxGlobal || maskLoc->nyGlobal != loc->nyGlobal) {
         if (mCommunicator->commRank() == 0) {
            ErrorLog(errorMessage);
            errorMessage.printf(
                  "%s: maskLayerName \"%s\" does not have the same x and y dimensions.\n",
                  getDescription_c(),
                  mMaskLayerName);
            errorMessage.printf(
                  "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                  maskLoc->nxGlobal,
                  maskLoc->nyGlobal,
                  maskLoc->nf,
                  loc->nxGlobal,
                  loc->nyGlobal,
                  loc->nf);
         }
         MPI_Barrier(mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }

      if (maskLoc->nf != 1 && maskLoc->nf != loc->nf) {
         if (mCommunicator->commRank() == 0) {
            ErrorLog(errorMessage);
            errorMessage.printf(
                  "%s: maskLayerName \"%s\" must either have the same number of features as this "
                  "layer, or one feature.\n",
                  getDescription_c(),
                  mMaskLayerName);
            errorMessage.printf(
                  "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                  maskLoc->nxGlobal,
                  maskLoc->nyGlobal,
                  maskLoc->nf,
                  loc->nxGlobal,
                  loc->nyGlobal,
                  loc->nf);
         }
         MPI_Barrier(mCommunicator->communicator());
         exit(EXIT_FAILURE);
      }
   }
   return Response::SUCCESS;
}

void MaskActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   ANNActivityBuffer::updateBufferCPU(simTime, deltaTime);

   float *A              = mBufferData.data();
   const PVLayerLoc *loc = getLayerLoc();

   int nx         = loc->nx;
   int ny         = loc->ny;
   int nf         = loc->nf;
   int numNeurons = nx * ny * nf;
   int nbatch     = loc->nbatch;

   for (int b = 0; b < nbatch; b++) {
      float *ABatch = A + b * getBufferSize();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int ni = 0; ni < numNeurons; ni++) {
         int kThisExt = kIndexExtended(
               ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         bool maskVal = true;

         switch (mMaskMethodCode) {
            case LAYER: {
               const PVLayerLoc *maskLoc      = mMaskBuffer->getLayerLoc();
               float const *maskActivity      = mMaskBuffer->getBufferData();
               float const *maskActivityBatch = maskActivity + b * mMaskBuffer->getBufferSize();
               int kMaskRes;
               if (maskLoc->nf == 1) {
                  kMaskRes = ni / nf;
               }
               else {
                  kMaskRes = ni;
               }
               int kMaskExt = kIndexExtended(
                     kMaskRes,
                     nx,
                     ny,
                     maskLoc->nf,
                     maskLoc->halo.lt,
                     maskLoc->halo.rt,
                     maskLoc->halo.dn,
                     maskLoc->halo.up);
               maskVal = maskActivityBatch[kMaskExt] ? true : false;
            } break;
            case INVERT_LAYER: {
               const PVLayerLoc *maskLoc      = mMaskBuffer->getLayerLoc();
               float const *maskActivity      = mMaskBuffer->getBufferData();
               float const *maskActivityBatch = maskActivity + b * mMaskBuffer->getBufferSize();
               int kMaskRes;
               if (maskLoc->nf == 1) {
                  kMaskRes = ni / nf;
               }
               else {
                  kMaskRes = ni;
               }
               int kMaskExt = kIndexExtended(
                     kMaskRes,
                     nx,
                     ny,
                     maskLoc->nf,
                     maskLoc->halo.lt,
                     maskLoc->halo.rt,
                     maskLoc->halo.dn,
                     maskLoc->halo.up);
               if (maskActivityBatch[kMaskExt]) {
                  maskVal = false;
               }
            } break;
            case FEATURES: {
               // Calculate feature index of ni
               int featureNum = featureIndex(ni, nx, ny, nf);
               maskVal        = true; // If nothing specified, copy everything
               for (int specF = 0; specF < mNumSpecifiedFeatures; specF++) {
                  if (featureNum == mFeatures[specF]) {
                     maskVal = false;
                     break;
                  }
               }
            } break;
            case INVERT_FEATURES: {
               // Calculate feature index of ni
               int featureNum = featureIndex(ni, nx, ny, nf);
               maskVal        = false; // If nothing specified, copy nothing
               for (int specF = 0; specF < mNumSpecifiedFeatures; specF++) {
                  if (featureNum == mFeatures[specF]) {
                     maskVal = true;
                     break;
                  }
               }
            } break;
            default: pvAssert(false);
         }

         // Set value to 0, otherwise, keep the value that ANNActivityBuffer computed
         if (maskVal == false) {
            ABatch[kThisExt] = 0.0f;
         }
      }
   }
}

} /* namespace PV */
