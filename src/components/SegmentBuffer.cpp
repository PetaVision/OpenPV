/*
 * SegmentBuffer.cpp
 *
 * created on: Jan 29, 2016
 *     Author: Sheng Lundquist
 */

#include "SegmentBuffer.hpp"

#include "components/ActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"

namespace PV {

SegmentBuffer::SegmentBuffer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

SegmentBuffer::SegmentBuffer() {}

void SegmentBuffer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

int SegmentBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_segmentMethod(ioFlag);
   return status;
}

void SegmentBuffer::ioParam_segmentMethod(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "segmentMethod", &segmentMethod);
   pvAssert(segmentMethod);
   // Check valid segment methods
   // none means the gsyn is already a segmentation. Helpful if reading segmentation from pvp
   if (strcmp(segmentMethod, "none") == 0) {
   }
   // TODO add in other segmentation methods
   // How do we segment across MPI margins?
   else {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s: segmentMethod %s not recognized. Current options are \"none\".\n",
               getDescription_c(),
               segmentMethod);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status
SegmentBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (!mOriginalActivity) {
      setOriginalActivity(message->mObjectTable);
   }
   pvAssert(mOriginalActivity);
   if (mOriginalActivity->getInitInfoCommunicatedFlag() == false) {
      return Response::POSTPONE;
   }

   checkDimensions();
   return Response::SUCCESS;
}

void SegmentBuffer::setOriginalActivity(ObserverTable const *table) {
   auto *originalLayerNameParam  = table->findObject<OriginalLayerNameParam>(getName());
   char const *originalLayerName = originalLayerNameParam->getLinkedObjectName();

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

void SegmentBuffer::checkDimensions() {
   // Original ActivityBuffer and SegmentBuffer must have same x- and y- dimensions
   ComponentBuffer::checkDimensionsXYEqual(mOriginalActivity, this);

   // SegmentBuffer must have nf == 1
   if (getLayerLoc()->nf != 1) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf("%s: SegmentLayer must have 1 feature.\n", getDescription_c());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }

   // If segmentMethod is none, we also need to make sure the srcLayer also has nf == 1
   if (strcmp(segmentMethod, "none") == 0 && mOriginalActivity->getLayerLoc()->nf != 1) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s: Source layer must have 1 feature with segmentation method \"none\".\n",
               getDescription_c());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

int SegmentBuffer::checkLabelBufSize(int newSize) {
   if (newSize <= mLabelBufSize) {
      return PV_SUCCESS;
   }

   // Grow buffer
   mLabelBuf = (int *)realloc(mLabelBuf, newSize * sizeof(int));
   mMaxXBuf  = (int *)realloc(mMaxXBuf, newSize * sizeof(int));
   mMaxYBuf  = (int *)realloc(mMaxYBuf, newSize * sizeof(int));
   mMinXBuf  = (int *)realloc(mMinXBuf, newSize * sizeof(int));
   mMinYBuf  = (int *)realloc(mMinYBuf, newSize * sizeof(int));

   // Set new size
   mLabelBufSize = newSize;
   return PV_SUCCESS;
}

int SegmentBuffer::loadLabelBuf() {
   // Load in maxX and label buf from maxX map
   int numLabels = mMaxX.size();
   // Allocate send buffer to the right size
   checkLabelBufSize(numLabels);

   int idx = 0;
   for (auto &m : mMaxX) {
      mLabelBuf[idx] = m.first; // Store key in label
      mMaxXBuf[idx]  = m.second; // Store vale in maxXBuf
      idx++;
   }
   pvAssert(idx == numLabels);

   // Load rest of buffers based on label
   for (int i = 0; i < numLabels; i++) {
      int label   = mLabelBuf[i];
      mMaxYBuf[i] = mMaxY.at(label);
      mMinXBuf[i] = mMinX.at(label);
      mMinYBuf[i] = mMinY.at(label);
   }
   return PV_SUCCESS;
}

int SegmentBuffer::loadCenterIdxMap(int batchIdx, int numLabels) {
   for (int i = 0; i < numLabels; i++) {
      int label                  = allLabelsBuf[i];
      int idx                    = centerIdxBuf[i];
      centerIdx[batchIdx][label] = idx;
   }
   return PV_SUCCESS;
}

int SegmentBuffer::checkIdxBufSize(int newSize) {
   if (newSize <= centerIdxBufSize) {
      return PV_SUCCESS;
   }

   // Grow buffer
   centerIdxBuf = (int *)realloc(centerIdxBuf, newSize * sizeof(int));
   allLabelsBuf = (int *)realloc(allLabelsBuf, newSize * sizeof(int));
   // Set new size
   centerIdxBufSize = newSize;
   return PV_SUCCESS;
}

Response::Status SegmentBuffer::allocateDataStructures() {
   auto status = ActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   int nbatch = getLayerLoc()->nbatch;
   mMaxX.clear();
   mMaxY.clear();
   mMinX.clear();
   mMinY.clear();
   centerIdx.clear();

   // Initialize vector of maps
   for (int b = 0; b < nbatch; b++) {
      centerIdx.push_back(std::map<int, int>());
   }

   return Response::SUCCESS;
}

void SegmentBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float const *origA = mOriginalActivity->getBufferData();
   float *thisA       = mBufferData.data();
   pvAssert(origA);
   pvAssert(thisA);

   const PVLayerLoc *loc = getLayerLoc();

   // Segment input layer based on segmentMethod
   if (strcmp(segmentMethod, "none") == 0) {
      int numBatchExtended = getBufferSizeAcrossBatch();
      // Copy activity over
      // Since both buffers should be identical size, we can do a memcpy here
      memcpy(thisA, origA, numBatchExtended * sizeof(float));
   }
   else {
      // This case should never happen
      pvAssert(0);
   }

   pvAssert(loc->nf == 1);

   // Clear centerIdxs
   for (int bi = 0; bi < loc->nbatch; bi++) {
      centerIdx[bi].clear();
   }

   for (int bi = 0; bi < loc->nbatch; bi++) {
      float *batchA = thisA + bi * getBufferSize();
      // Reset max/min buffers
      mMaxX.clear();
      mMaxY.clear();
      mMinX.clear();
      mMinY.clear();

      // Loop through this buffer to fill labelVec and idxVec
      // Looping through restricted, but indices are extended
      for (int yi = loc->halo.up; yi < loc->ny + loc->halo.up; yi++) {
         for (int xi = loc->halo.lt; xi < loc->nx + loc->halo.lt; xi++) {
            // Convert to local extended linear index
            int niLocalExt = yi * (loc->nx + loc->halo.lt + loc->halo.rt) + xi;
            // Convert yi and xi to global res index
            int globalResYi = yi - loc->halo.up + loc->ky0;
            int globalResXi = xi - loc->halo.lt + loc->kx0;

            // Get label value
            // Note that we're assuming that the activity here are integers,
            // even though the buffer is floats
            int labelVal = round(batchA[niLocalExt]);

            // Calculate max/min x and y for a single batch
            // If labelVal exists in map
            if (mMaxX.count(labelVal)) {
               // Here, we're assuming the 4 maps are in sync, so we use the
               //.at method, as it will throw an exception as opposed to the
               //[] operator, which will simply add the key into the map
               if (globalResXi > mMaxX.at(labelVal)) {
                  mMaxX[labelVal] = globalResXi;
               }
               if (globalResXi < mMinX.at(labelVal)) {
                  mMinX[labelVal] = globalResXi;
               }
               if (globalResYi > mMaxY.at(labelVal)) {
                  mMaxY[labelVal] = globalResYi;
               }
               if (globalResYi < mMinY.at(labelVal)) {
                  mMinY[labelVal] = globalResYi;
               }
            }
            // If doesn't exist, add into map with current vals
            else {
               mMaxX[labelVal] = globalResXi;
               mMinX[labelVal] = globalResXi;
               mMaxY[labelVal] = globalResYi;
               mMinY[labelVal] = globalResYi;
            }
         }
      }

      // We need to mpi across processors in case a segment crosses an mpi boundary
      Communicator const *icComm = mCommunicator;
      int numMpi                 = icComm->commSize();
      int rank                   = icComm->commRank();

      // Local comm rank
      // Non root processes simply send buffer size and then buffers
      int numLabels = mMaxX.size();

      if (rank != 0) {
         // Load buffers
         loadLabelBuf();
         // Send number of labels first
         MPI_Send(&numLabels, 1, MPI_INT, 0, rank, icComm->communicator());
         // Send labels, then max/min buffers
         MPI_Send(mLabelBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(mMaxXBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(mMaxYBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(mMinXBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());
         MPI_Send(mMinYBuf, numLabels, MPI_INT, 0, rank, icComm->communicator());

         // Receive the full centerIdxBuf from root process
         int numCenterIdx = 0;
         MPI_Bcast(&numCenterIdx, 1, MPI_INT, 0, icComm->communicator());
         checkIdxBufSize(numCenterIdx);

         MPI_Bcast(allLabelsBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());
         MPI_Bcast(centerIdxBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());

         // Load buffer into centerIdx map
         loadCenterIdxMap(bi, numCenterIdx);
      }
      // Root process stores everything
      else {
         // One recv per buffer
         for (int recvRank = 1; recvRank < numMpi; recvRank++) {
            int numRecvLabels = 0;
            MPI_Recv(&numRecvLabels, 1, MPI_INT, recvRank, recvRank, icComm->communicator(), NULL);
            checkLabelBufSize(numRecvLabels);

            MPI_Recv(
                  mLabelBuf,
                  numRecvLabels,
                  MPI_INT,
                  recvRank,
                  recvRank,
                  icComm->communicator(),
                  NULL);
            MPI_Recv(
                  mMaxXBuf,
                  numRecvLabels,
                  MPI_INT,
                  recvRank,
                  recvRank,
                  icComm->communicator(),
                  NULL);
            MPI_Recv(
                  mMaxYBuf,
                  numRecvLabels,
                  MPI_INT,
                  recvRank,
                  recvRank,
                  icComm->communicator(),
                  NULL);
            MPI_Recv(
                  mMinXBuf,
                  numRecvLabels,
                  MPI_INT,
                  recvRank,
                  recvRank,
                  icComm->communicator(),
                  NULL);
            MPI_Recv(
                  mMinYBuf,
                  numRecvLabels,
                  MPI_INT,
                  recvRank,
                  recvRank,
                  icComm->communicator(),
                  NULL);

            for (int i = 0; i < numRecvLabels; i++) {
               int label = mLabelBuf[i];
               // Add on to maps
               // If the label already exists, fill with proper max/min
               if (mMaxX.count(label)) {
                  if (mMaxXBuf[i] > mMaxX.at(label)) {
                     mMaxX[label] = mMaxXBuf[i];
                  }
                  if (mMaxYBuf[i] > mMaxY.at(label)) {
                     mMaxY[label] = mMaxYBuf[i];
                  }
                  if (mMinXBuf[i] < mMinX.at(label)) {
                     mMinX[label] = mMinXBuf[i];
                  }
                  if (mMinYBuf[i] < mMinY.at(label)) {
                     mMinY[label] = mMinYBuf[i];
                  }
               }
               else {
                  mMaxX[label] = mMaxXBuf[i];
                  mMaxY[label] = mMaxYBuf[i];
                  mMinX[label] = mMinXBuf[i];
                  mMinY[label] = mMinYBuf[i];
               }
            }
         }

         // Maps are now filled with all segments from the image
         // Fill centerIdx based on max/min
         for (auto &m : mMaxX) {
            int label   = m.first;
            int centerX = mMinX.at(label) + (mMaxX.at(label) - mMinX.at(label)) / 2;
            int centerY = mMinY.at(label) + (mMaxY.at(label) - mMinY.at(label)) / 2;
            // Convert centerpoints (in global res idx) to linear idx (in global res space)
            int centerIdxVal = centerY * (loc->nxGlobal) + centerX;
            // Add to centerIdxMap
            centerIdx[bi][label] = centerIdxVal;
         }

         // Fill centerpoint buffer
         int numCenterIdx = centerIdx[bi].size();
         checkIdxBufSize(numCenterIdx);

         int idx = 0;
         for (auto &ctr : centerIdx[bi]) {
            allLabelsBuf[idx] = ctr.first;
            centerIdxBuf[idx] = ctr.second;
            idx++;
         }

         // Broadcast buffers
         MPI_Bcast(&numCenterIdx, 1, MPI_INT, 0, icComm->communicator());
         MPI_Bcast(allLabelsBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());
         MPI_Bcast(centerIdxBuf, numCenterIdx, MPI_INT, 0, icComm->communicator());
      }
   } // End batch loop

   // centerIdx now stores each center coordinate of each segment
}

SegmentBuffer::~SegmentBuffer() {
   mMaxX.clear();
   mMaxY.clear();
   mMinX.clear();
   mMinY.clear();
   // This should call destructors of all maps within the vector
   centerIdx.clear();
}

} /* namespace PV */
