/*
 * BackgroundActivityBuffer.cpp
 *
 *  Created on: 4/16/15
 *  slundquist
 */

#include "BackgroundActivityBuffer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

BackgroundActivityBuffer::BackgroundActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

BackgroundActivityBuffer::~BackgroundActivityBuffer() {}

void BackgroundActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

void BackgroundActivityBuffer::setObjectType() { mObjectType = "BackgroundActivityBuffer"; }

int BackgroundActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ActivityBuffer::ioParamsFillGroup(ioFlag);
   ioParam_repFeatureNum(ioFlag);
   return status;
}

void BackgroundActivityBuffer::ioParam_repFeatureNum(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "repFeatureNum", &mRepFeatureNum, mRepFeatureNum);
   if (mRepFeatureNum <= 0) {
      Fatal() << "BackgroundLayer " << name << ": repFeatureNum must an integer greater or equal "
                                               "to 1 (1 feature means no replication)\n";
   }
}

Response::Status BackgroundActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *objectTable            = message->mObjectTable;
   auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
   FatalIf(
         originalLayerNameParam == nullptr,
         "%s could not find an OriginalLayerNameParam component.\n",
         getDescription_c());
   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   char const *originalLayerName = originalLayerNameParam->getLinkedObjectName();
   mOriginalData = objectTable->findObject<BasePublisherComponent>(originalLayerName);
   FatalIf(
         mOriginalData == nullptr,
         "%s originalLayerName \"%s\" does not have a BasePublisherComponent.\n",
         getDescription_c(),
         originalLayerNameParam->getLinkedObjectName());
   checkDimensions();
   return Response::SUCCESS;
}

void BackgroundActivityBuffer::checkDimensions() const {
   PVLayerLoc const *locOriginal = mOriginalData->getLayerLoc();
   PVLayerLoc const *loc         = getLayerLoc();
   FatalIf(
         locOriginal->nbatch != loc->nbatch,
         "%s and %s do not have the same batch width (%d versus %d)\n",
         mOriginalData->getDescription_c(),
         getDescription_c(),
         locOriginal->nbatch,
         loc->nbatch);

   bool dimsEqual = true;
   dimsEqual      = dimsEqual and (locOriginal->nx == loc->nx);
   dimsEqual      = dimsEqual and (locOriginal->ny == loc->ny);
   FatalIf(
         !dimsEqual,
         "%s and %s do not have the same x- and y- dimensions (%d-by-%d) versus (%d-by-%d).\n",
         mOriginalData->getDescription_c(),
         getDescription_c(),
         locOriginal->nx,
         locOriginal->nx,
         loc->nx,
         loc->ny);

   int const origNumFeatures    = locOriginal->nf;
   int const correctNumFeatures = (origNumFeatures + 1) * mRepFeatureNum;
   FatalIf(
         loc->nf != correctNumFeatures,
         "The parameters of %s are not consistent with the number of features of %s."
         "The original layer has n=%d features; therefore the background layer must have "
         "(n+1)*repFeatureNum = (%d+1)*%d = %d features instead of %d.\n",
         getDescription_c(),
         mOriginalData->getDescription_c(),
         locOriginal->nf,
         locOriginal->nf,
         mRepFeatureNum,
         correctNumFeatures,
         loc->nf);
}

Response::Status
BackgroundActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   int const numExtendedAcrossBatch = getBufferSizeAcrossBatch();
   float *activityData              = getReadWritePointer();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int kExt = 0; kExt < numExtendedAcrossBatch; kExt++) {
      activityData[kExt] = 0.0f;
   }

   return Response::SUCCESS;
}

void BackgroundActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *A                      = mBufferData.data();
   float const *originalA        = mOriginalData->getLayerData(0);
   PVLayerLoc const *loc         = getLayerLoc();
   PVLayerLoc const *locOriginal = mOriginalData->getLayerLoc();

   // Make sure all sizes match (this was checked in checkDimensions)
   assert(locOriginal->nx == loc->nx);
   assert(locOriginal->ny == loc->ny);
   assert((locOriginal->nf + 1) * mRepFeatureNum == loc->nf);

   int nx     = loc->nx;
   int ny     = loc->ny;
   int origNf = locOriginal->nf;
   int thisNf = loc->nf;
   int nbatch = loc->nbatch;

   int originalNxExtended  = locOriginal->nx + locOriginal->halo.lt + locOriginal->halo.rt;
   int originalNyExtended  = locOriginal->ny + locOriginal->halo.dn + locOriginal->halo.up;
   int originalNumExtended = originalNxExtended * originalNyExtended * locOriginal->nf;

   for (int b = 0; b < nbatch; b++) {
      float *ABatch               = A + b * getBufferSize();
      float const *originalABatch = originalA + b * originalNumExtended;

// Loop through all nx and ny
// each y value specifies a different target so ok to thread here (sum, sumsq are defined inside
// loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int iY = 0; iY < ny; iY++) {
         for (int iX = 0; iX < nx; iX++) {
            // outVal stores the NOR of the other values
            int outVal = 1;
            // Shift all features down by one
            for (int iF = 0; iF < origNf; iF++) {
               int kextOrig = kIndex(
                     iX,
                     iY,
                     iF,
                     nx + locOriginal->halo.lt + locOriginal->halo.rt,
                     ny + locOriginal->halo.dn + locOriginal->halo.up,
                     origNf);
               float origActivity = originalABatch[kextOrig];
               // outVal is the final out value for the background
               if (origActivity != 0) {
                  outVal = 0;
               }
               // Loop over replicated features
               for (int repIdx = 0; repIdx < mRepFeatureNum; repIdx++) {
                  // Index iF one down, multiply by replicate feature number, add repIdx offset
                  int newFeatureIdx = ((iF + 1) * mRepFeatureNum) + repIdx;
                  assert(newFeatureIdx < thisNf);
                  int kext = kIndex(
                        iX,
                        iY,
                        newFeatureIdx,
                        nx + loc->halo.lt + loc->halo.rt,
                        ny + loc->halo.dn + loc->halo.up,
                        thisNf);
                  ABatch[kext] = origActivity;
               }
            }
            // Set background indices to outVal
            for (int repIdx = 0; repIdx < mRepFeatureNum; repIdx++) {
               int kextBackground = kIndex(
                     iX,
                     iY,
                     repIdx,
                     nx + loc->halo.lt + loc->halo.rt,
                     ny + loc->halo.dn + loc->halo.up,
                     thisNf);
               ABatch[kextBackground] = outVal;
            }
         }
      }
   }
}

} // namespace PV
