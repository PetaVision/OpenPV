/*
 * SegmentifyBuffer.cpp
 *
 * created on: Feb 10, 2016
 *     Author: Sheng Lundquist
 */

#include "SegmentifyBuffer.hpp"

#include "components/ActivityComponent.hpp"
#include "components/SegmentBuffer.hpp"

namespace PV {

SegmentifyBuffer::SegmentifyBuffer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

SegmentifyBuffer::SegmentifyBuffer() {
   // initialize() gets called by subclass's initialize method
}

SegmentifyBuffer::~SegmentifyBuffer() {}

void SegmentifyBuffer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

int SegmentifyBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_segmentLayerName(ioFlag);
   ioParam_inputMethod(ioFlag);
   ioParam_outputMethod(ioFlag);
   return status;
}

void SegmentifyBuffer::ioParam_inputMethod(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "inputMethod", &mInputMethod);
   if (strcmp(mInputMethod, "average") == 0) {
   }
   else if (strcmp(mInputMethod, "sum") == 0) {
   }
   else if (strcmp(mInputMethod, "max") == 0) {
   }
   else {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s: inputMethod must be \"average\", \"sum\", or \"max\".\n", getDescription_c());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

void SegmentifyBuffer::ioParam_outputMethod(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "outputMethod", &mOutputMethod);
   if (strcmp(mOutputMethod, "centroid") == 0) {
   }
   else if (strcmp(mOutputMethod, "fill") == 0) {
   }
   else {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s: outputMethod must be \"centriod\" or \"fill\".\n", getDescription_c());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

void SegmentifyBuffer::ioParam_segmentLayerName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "segmentLayerName", &mSegmentLayerName);
   assert(mSegmentLayerName);
   if (ioFlag == PARAMS_IO_READ && mSegmentLayerName[0] == '\0') {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf("%s: segmentLayerName must be set.\n", getDescription_c());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status
SegmentifyBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   ObserverTable const *objectTable = message->mObjectTable;

   if (!mOriginalActivity) {
      setOriginalActivity(objectTable);
   }
   pvAssert(mOriginalActivity);
   if (mOriginalActivity->getInitInfoCommunicatedFlag() == false) {
      return Response::POSTPONE;
   }

   if (!mSegmentBuffer) {
      setSegmentBuffer(objectTable);
   }
   pvAssert(mSegmentBuffer);
   if (mSegmentBuffer->getInitInfoCommunicatedFlag() == false) {
      return Response::POSTPONE;
   }

   checkDimensions();
   return Response::SUCCESS;
}

void SegmentifyBuffer::setOriginalActivity(ObserverTable const *table) {
   auto *originalLayerNameParam = table->findObject<OriginalLayerNameParam>(getName());
   FatalIf(
         originalLayerNameParam == nullptr,
         "%s does not have an OriginalLayerNameParam.\n",
         getDescription_c());

   char const *originalLayerName = originalLayerNameParam->getName();

   // Sync margins
   auto *origGeometry = table->findObject<LayerGeometry>(originalLayerName);
   auto *thisGeometry = table->findObject<LayerGeometry>(getName());
   LayerGeometry::synchronizeMarginWidths(thisGeometry, origGeometry);

   // Get original layer's activity buffer
   mOriginalActivity = table->findObject<ActivityBuffer>(originalLayerName);
   FatalIf(
         mOriginalActivity == nullptr,
         "%s: no object named \"%s\" with an ActivityBuffer.\n",
         getDescription_c());
}

void SegmentifyBuffer::setSegmentBuffer(ObserverTable const *table) {
   // Get SegmentBuffer from segment layer
   mSegmentBuffer = table->findObject<SegmentBuffer>(mSegmentLayerName);
   FatalIf(
         mSegmentBuffer == nullptr,
         "%s could not find a SegmentBuffer within segment layer \"%s\".\n",
         getDescription_c(),
         mSegmentLayerName);
}

void SegmentifyBuffer::checkDimensions() {
   // Check sizes
   const PVLayerLoc *origLoc = mOriginalActivity->getLayerLoc();
   const PVLayerLoc *thisLoc = getLayerLoc();
   pvAssert(origLoc != nullptr && thisLoc != nullptr);

   // Original layer must have the same number of features as this layer
   // The x- and y- dimensions can differ.
   if (origLoc->nf != thisLoc->nf) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayer \"%s\" does not have the same feature dimension as this layer.\n",
               getDescription_c(),
               mOriginalActivity->getName());
         errorMessage.printf("    original (nf=%d) versus (nf=%d)\n", origLoc->nf, thisLoc->nf);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }

   // Segment layer must have nf==1 (already checked by SegmentBuffer::checkDimensions() method).
   pvAssert(mSegmentBuffer->getLayerLoc() and mSegmentBuffer->getLayerLoc()->nf == 1);
}

Response::Status SegmentifyBuffer::allocateDataStructures() {
   auto status = ActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   mLabelToIdx.clear();
   mLabelVals   = (float **)calloc(getLayerLoc()->nf, sizeof(float *));
   mLabelCount  = (int **)calloc(getLayerLoc()->nf, sizeof(int *));
   mLabelIdxBuf = NULL;
   // Don't allocate inner buffers yet; this will get done based on how many labels are in the
   // current image

   return Response::SUCCESS;
}

void SegmentifyBuffer::updateBufferCPU(double simTime, double deltaTime) {
   // Using the segment activity, we want to compress all values within a segment to a single value
   // (per feature)
   for (int bi = 0; bi < getLayerLoc()->nbatch; bi++) {
      buildLabelToIdx(bi);
      calculateLabelVals(bi);
      setOutputVals(bi);
   }
}

void SegmentifyBuffer::checkLabelValBuf(int newSize) {
   if (newSize <= mNumLabelVals) {
      return;
   }

   // Grow buffers
   for (int i = 0; i < getLayerLoc()->nf; i++) {
      mLabelVals[i]  = (float *)realloc(mLabelVals[i], newSize * sizeof(float));
      mLabelCount[i] = (int *)realloc(mLabelCount[i], newSize * sizeof(int));
   }
   mLabelIdxBuf = (int *)realloc(mLabelIdxBuf, newSize * sizeof(int));

   mNumLabelVals = newSize;
}

void SegmentifyBuffer::buildLabelToIdx(int batchIdx) {
   Communicator const *icComm = mCommunicator;
   int rank                   = icComm->commRank();

   mLabelToIdx.clear();
   // First, we need a single scalar per feature per segment label
   // We need to build a data structure that maps from labels to a vector index
   int numLabels = 0;
   if (rank == 0) {
      std::map<int, int> segMap = mSegmentBuffer->getCenterIdxBuf(batchIdx);
      // From the map, we want to grab the set of keys and store it into an int array for
      // broadcasting
      numLabels = segMap.size();
      // Adjust size of buffers
      checkLabelValBuf(numLabels);
      // Fill buffer
      int l = 0;
      for (auto &seg : segMap) {
         mLabelIdxBuf[l] = seg.first;
         l++;
      }
   }

   // Broadcast number and list of labels from the root process to rest
   MPI_Bcast(&numLabels, 1, MPI_INT, 0, icComm->communicator());
   checkLabelValBuf(numLabels);
   MPI_Bcast(mLabelIdxBuf, numLabels, MPI_INT, 0, icComm->communicator());

   for (int l = 0; l < numLabels; l++) {
      // Translate the label buffer into the labelToIdx buffer
      mLabelToIdx[mLabelIdxBuf[l]] = l;
      // Initialize labelVals based on value reduction type
      // If max, initialize to -inf
      for (int fi = 0; fi < getLayerLoc()->nf; fi++) {
         // Set count to 0
         mLabelCount[fi][l] = 0;
         if (strcmp(mInputMethod, "max") == 0) {
            mLabelVals[fi][l] = -INFINITY;
         }
         // If average or sum, initialize to 0
         else if (strcmp(mInputMethod, "average") == 0 || strcmp(mInputMethod, "sum") == 0) {
            mLabelVals[fi][l] = 0;
         }
         else {
            assert(0); // should never get here
         }
      }
   }
}

void SegmentifyBuffer::calculateLabelVals(int batchIdx) {
   Communicator const *icComm = mCommunicator;

   const PVLayerLoc *srcLoc = mOriginalActivity->getLayerLoc();
   const PVLayerLoc *segLoc = mSegmentBuffer->getLayerLoc();

   assert(segLoc->nf == 1);

   float const *srcA = mOriginalActivity->getBufferData();
   float const *segA = mSegmentBuffer->getBufferData();

   assert(srcA);
   assert(segA);

   float const *srcBatchA = srcA + batchIdx * mOriginalActivity->getBufferSize();
   float const *segBatchA = segA + batchIdx * mSegmentBuffer->getBufferSize();

   // Loop through source values
   // As segments are restricted only, we loop through restricted activity
   for (int yi = 0; yi < srcLoc->ny; yi++) {
      // We caluclate the index into the segment buffer and this buffer based on the
      // relative size differences between source and label buffers
      float segToSrcScaleY = (float)segLoc->ny / (float)srcLoc->ny;
      int segmentYi        = round(yi * segToSrcScaleY);
      for (int xi = 0; xi < srcLoc->nx; xi++) {
         float segToSrcScaleX = (float)segLoc->nx / (float)srcLoc->nx;
         int segmentXi        = round(xi * segToSrcScaleX);
         // Convert segment x and y index into extended linear index into the segment buffer
         int extSegIdx =
               (segmentYi + segLoc->halo.up) * (segLoc->nx + segLoc->halo.lt + segLoc->halo.rt)
               + (segmentXi + segLoc->halo.lt);

         // Assuming segments are ints
         int labelVal = round(segBatchA[extSegIdx]);

         // This label should always exist in the map
         // labelIdx is the index into the vals buffer
         int labelIdx = mLabelToIdx.at(labelVal);

         for (int fi = 0; fi < srcLoc->nf; fi++) {
            // Convert restricted yi and xi to extended
            // with resepct to the source
            int extSrcIdx = (yi + srcLoc->halo.up)
                                  * (srcLoc->nx + srcLoc->halo.lt + srcLoc->halo.rt) * srcLoc->nf
                            + (xi + srcLoc->halo.lt) * srcLoc->nf + fi;
            float srcVal = srcBatchA[extSrcIdx];
            mLabelCount[fi][labelIdx]++;
            // Fill labelVals and labelCount
            if (strcmp(mInputMethod, "max") == 0) {
               if (mLabelVals[fi][labelIdx] < srcVal) {
                  mLabelVals[fi][labelIdx] = srcVal;
               }
            }
            else if (strcmp(mInputMethod, "average") == 0 || strcmp(mInputMethod, "sum") == 0) {
               mLabelVals[fi][labelIdx] += srcVal;
            }
         } // End of fi loop
      } // End of xi loop
   } // End of yi loop

   int numLabels = mLabelToIdx.size();

   // We need to reduce our labelVec array
   for (int fi = 0; fi < srcLoc->nf; fi++) {
      MPI_Allreduce(
            MPI_IN_PLACE, mLabelCount[fi], numLabels, MPI_INT, MPI_SUM, icComm->communicator());
      if (strcmp(mInputMethod, "max") == 0) {
         MPI_Allreduce(
               MPI_IN_PLACE, mLabelVals[fi], numLabels, MPI_FLOAT, MPI_MAX, icComm->communicator());
      }
      else if (strcmp(mInputMethod, "sum") == 0 || strcmp(mInputMethod, "average") == 0) {
         MPI_Allreduce(
               MPI_IN_PLACE, mLabelVals[fi], numLabels, MPI_FLOAT, MPI_SUM, icComm->communicator());
      }
      // If average, divide sum by count
      if (strcmp(mInputMethod, "average") == 0) {
         for (int l = 0; l < numLabels; l++) {
            mLabelVals[fi][l] = mLabelVals[fi][l] / mLabelCount[fi][l];
         }
      }
   }
}

void SegmentifyBuffer::setOutputVals(int batchIdx) {
   // Given the labelVals, we want to fill the output A buffer with what each val should be
   const PVLayerLoc *segLoc  = mSegmentBuffer->getLayerLoc();
   const PVLayerLoc *thisLoc = getLayerLoc();

   assert(segLoc->nf == 1);

   float const *segA = mSegmentBuffer->getBufferData();
   float *thisA      = mBufferData.data();

   assert(thisA);
   assert(segA);

   float const *segBatchA = segA + batchIdx * mSegmentBuffer->getBufferSize();
   float *thisBatchA      = thisA + batchIdx * getBufferSize();

   // Reset activity values
   for (int ni = 0; ni < getBufferSize(); ni++) {
      thisBatchA[ni] = 0;
   }

   // Scale factors between this layer and segment layer
   float thisToSegScaleX = (float)thisLoc->nx / (float)segLoc->nx;
   float thisToSegScaleY = (float)thisLoc->ny / (float)segLoc->ny;

   // If by centroid, get centroid map from SegmentLayer and set each value
   if (strcmp(mOutputMethod, "centroid") == 0) {
      std::map<int, int> segMap = mSegmentBuffer->getCenterIdxBuf(batchIdx);
      // Centroids are stored in global restricted space, with respect to the segment layer
      for (auto &seg : segMap) {
         int label           = seg.first;
         int segGlobalResIdx = seg.second;
         // Convert to restrictd x and y coords wrt segment layer
         int segGlobalResX = segGlobalResIdx % (segLoc->nxGlobal);
         int segGlobalResY = segGlobalResIdx / (segLoc->nyGlobal);
         // Convert to x and y wrt this layer
         int thisGlobalResX = round(segGlobalResX * thisToSegScaleX);
         int thisGlobalResY = round(segGlobalResY * thisToSegScaleY);
         // If we're within bounds in this process
         if (thisGlobalResX >= thisLoc->kx0 && thisGlobalResX < thisLoc->kx0 + thisLoc->nx
             && thisGlobalResY >= thisLoc->ky0
             && thisGlobalResY < thisLoc->ky0 + thisLoc->ny) {
            // Convert thisGlobalResX and Y to an extended local linear index
            int thisLocalExtX = thisGlobalResX - thisLoc->kx0 + thisLoc->halo.lt;
            int thisLocalExtY = thisGlobalResY - thisLoc->ky0 + thisLoc->halo.up;
            for (int fi = 0; fi < thisLoc->nf; fi++) {
               int thisLocalExtIdx = thisLocalExtY
                                           * (thisLoc->nx + thisLoc->halo.lt + thisLoc->halo.rt)
                                           * thisLoc->nf
                                     + thisLocalExtX * thisLoc->nf + fi;
               // Set value based on labelVals
               thisBatchA[thisLocalExtIdx] = mLabelVals[fi][mLabelToIdx.at(label)];
            }
         }
      }
   }
   else if (strcmp(mOutputMethod, "fill") == 0) {
      // Loop through this layer's neurons
      // Looping through restricted
      for (int yi = 0; yi < thisLoc->ny; yi++) {
         // Translate from this yi to segment's yi
         int segResY = round((float)yi / (float)thisToSegScaleY);
         for (int xi = 0; xi < thisLoc->nx; xi++) {
            int segResX = round((float)xi / (float)thisToSegScaleX);
            // Convert restricted segment index to extended
            int segExtIdx =
                  (segResY + segLoc->halo.up) * (segLoc->nx + segLoc->halo.lt + segLoc->halo.rt)
                  + (segResX + segLoc->halo.lt);
            // Get label based on segment layer
            int label = round(segBatchA[segExtIdx]);
            // Fill index with value from labelVals;
            for (int fi = 0; fi < thisLoc->nf; fi++) {
               // Calulate ext index
               int thisExtIdx = (yi + thisLoc->halo.up)
                                      * (thisLoc->nx + thisLoc->halo.lt + thisLoc->halo.rt)
                                      * thisLoc->nf
                                + (xi + thisLoc->halo.lt) * thisLoc->nf + fi;
               thisBatchA[thisExtIdx] = mLabelVals[fi][mLabelToIdx.at(label)];
            }
         }
      }
   }
}

} /* namespace PV */
