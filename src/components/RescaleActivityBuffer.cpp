/*
 * RescaleActivityBuffer.cpp
 */

#include "RescaleActivityBuffer.hpp"

#include "components/OriginalLayerNameParam.hpp"

#include "arch/mpi/mpi.h"                         // for MPI_Allreduce, MPI_...
#include "cMakeHeader.h"                          // for PV_USE_OPENMP_THREADS
#include "components/ActivityBuffer.hpp"          // for ActivityBuffer
#include "components/OriginalLayerNameParam.hpp"  // for OriginalLayerNameParam
#include "structures/PVLayerLoc.hpp"                 // for PVLayerLoc, PVHalo
#include "include/pv_common.h"                    // for PV_SUCCESS
#include "observerpattern/ObserverTable.hpp"      // for ObserverTable
#include "utils/PVAssert.hpp"                     // for pvAssert
#include "utils/PVLog.hpp"                        // for Fatal, Log, FatalIf
#include "utils/conversions.hpp"                  // for kIndexExtended, kIndex
#include <algorithm>                              // for max
#include <cfloat>                                 // for FLT_MIN
#include <cmath>                                  // for sqrt, exp
#include <cstdlib>                                // for free
#include <cstring>                                // for strcmp, memset
#include <vector>                                 // for vector

namespace PV {
RescaleActivityBuffer::RescaleActivityBuffer() {}

RescaleActivityBuffer::RescaleActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

RescaleActivityBuffer::~RescaleActivityBuffer() {}

void RescaleActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ActivityBuffer::initialize(paramsIO, comm);
}

// This is almost an exact duplicate of CloneInternalStateBuffer::communicateInitInfo; a separate
// method is necessary because CloneISB is an InternalStateBuffer, but RescaleActivityBuffer is
// an ActivityBuffer.
Response::Status RescaleActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (mOriginalBuffer == nullptr) {
      auto *objectTable            = message->mObjectTable;
      auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
      if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }
      FatalIf(
            originalLayerNameParam == nullptr,
            "%s could not find an OriginalLayerNameParam.\n",
            getDescription_c());

      // Retrieve original layer's ActivityBuffer
      std::string const &originalLayerName = originalLayerNameParam->getLinkedObjectName();
      mOriginalBuffer               = objectTable->findObject<ActivityBuffer>(originalLayerName);
      FatalIf(
            mOriginalBuffer == nullptr,
            "%s could not find an ActivityBuffer within %s.\n",
            getDescription_c(),
            originalLayerName.c_str());
   }
   if (!mOriginalBuffer->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   checkDimensionsEqual(mOriginalBuffer, this);

   return Response::SUCCESS;
}

int RescaleActivityBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ActivityBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_rescaleMethod(ioSwitch);
   if (mRescaleMethod == "maxmin") {
      mMethodCode = MAXMIN;
      ioParam_targetMax(ioSwitch);
      ioParam_targetMin(ioSwitch);
   }
   else if (mRescaleMethod == "meanstd") {
      mMethodCode = MEANSTD;
      ioParam_targetMean(ioSwitch);
      ioParam_targetStd(ioSwitch);
   }
   else if (mRescaleMethod == "pointmeanstd") {
      mMethodCode = POINTMEANSTD;
      ioParam_targetMean(ioSwitch);
      ioParam_targetStd(ioSwitch);
   }
   else if (mRescaleMethod == "l2") {
      mMethodCode = L2;
      ioParam_patchSize(ioSwitch);
   }
   else if (mRescaleMethod == "l2NoMean") {
      mMethodCode = L2NOMEAN;
      ioParam_patchSize(ioSwitch);
   }
   else if (mRescaleMethod == "pointResponseNormalization") {
      mMethodCode = POINTRESPONSENORMALIZATION;
   }
   else if (mRescaleMethod == "zerotonegative") {
      mMethodCode = ZEROTONEGATIVE;
   }
   else if (mRescaleMethod == "softmax") {
      mMethodCode = SOFTMAX;
   }
   else if (mRescaleMethod == "logreg") {
      mMethodCode = LOGREG;
   }
   else {
      Fatal().printf(
            "%s: rescaleMethod does not exist. Current implemented methods are "
            "maxmin, meanstd, pointmeanstd, pointResponseNormalization, softmax, l2, l2NoMean, "
            "zerotonegative, and logreg.\n",
            getDescription_c());
   }
   return PV_SUCCESS;
}

void RescaleActivityBuffer::ioParam_rescaleMethod(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "rescaleMethod", &mRescaleMethod);
}

void RescaleActivityBuffer::ioParam_targetMax(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("rescaleMethod"));
   if (mMethodCode == MAXMIN) {
      mParamsIO->ioParam(ioSwitch, "targetMax", &mTargetMax);
   }
}

void RescaleActivityBuffer::ioParam_targetMin(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("rescaleMethod"));
   if (mMethodCode == MAXMIN) {
      mParamsIO->ioParam(ioSwitch, "targetMin", &mTargetMin);
   }
}

void RescaleActivityBuffer::ioParam_targetMean(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("rescaleMethod"));
   if (mMethodCode == MEANSTD or mMethodCode == POINTMEANSTD) {
      mParamsIO->ioParam(ioSwitch, "targetMean", &mTargetMean);
   }
}

void RescaleActivityBuffer::ioParam_targetStd(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("rescaleMethod"));
   if (mMethodCode == MEANSTD or mMethodCode == POINTMEANSTD) {
      mParamsIO->ioParam(ioSwitch, "targetStd", &mTargetStd);
   }
}

void RescaleActivityBuffer::ioParam_patchSize(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("rescaleMethod"));
   if (mMethodCode == L2 or mMethodCode == L2NOMEAN) {
      mParamsIO->ioParam(ioSwitch, "patchSize", &mPatchSize);
   }
}

Response::Status
RescaleActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   float *A = mBufferData.data();
   std::memset(A, 0, sizeof(float) * getBufferSizeAcrossBatch());
   return Response::SUCCESS;
}

// GTK: changed to rescale activity instead of V
void RescaleActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   float *A                      = mBufferData.data();
   float const *originalA        = mOriginalBuffer->getBufferData();
   const PVLayerLoc *loc         = getLayerLoc();
   int const numNeurons          = loc->nx * loc->ny * loc->nf;
   int const numGlobalNeurons    = loc->nxGlobal * loc->nyGlobal * loc->nf;
   const PVLayerLoc *locOriginal = mOriginalBuffer->getLayerLoc();
   int nbatch                    = loc->nbatch;
   // Make sure all sizes match (this was checked in communicateInitInfo)
   pvAssert(locOriginal->nx == loc->nx);
   pvAssert(locOriginal->ny == loc->ny);
   pvAssert(locOriginal->nf == loc->nf);

   for (int b = 0; b < nbatch; b++) {
      float const *originalABatch = originalA + b * mOriginalBuffer->getBufferSize();
      float *ABatch               = A + b * getBufferSize();

      if (mMethodCode == MAXMIN) {
         float maxA = -1000000000;
         float minA = 1000000000;
         // Find max and min of A
         for (int k = 0; k < numNeurons; k++) {
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  locOriginal->halo.rt,
                  locOriginal->halo.dn,
                  locOriginal->halo.up);
            if (originalABatch[kextOriginal] > maxA) {
               maxA = originalABatch[kextOriginal];
            }
            if (originalABatch[kextOriginal] < minA) {
               minA = originalABatch[kextOriginal];
            }
         }

         MPI_Allreduce(MPI_IN_PLACE, &maxA, 1, MPI_FLOAT, MPI_MAX, mCommunicator->communicator());
         MPI_Allreduce(MPI_IN_PLACE, &minA, 1, MPI_FLOAT, MPI_MIN, mCommunicator->communicator());

         float rangeA = maxA - minA;
         if (rangeA != 0) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
            for (int k = 0; k < numNeurons; k++) {
               int kExt = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.dn,
                     loc->halo.up);
               int kExtOriginal = kIndexExtended(
                     k,
                     locOriginal->nx,
                     locOriginal->ny,
                     locOriginal->nf,
                     locOriginal->halo.lt,
                     locOriginal->halo.rt,
                     locOriginal->halo.dn,
                     locOriginal->halo.up);
               ABatch[kExt] =
                     ((originalABatch[kExtOriginal] - minA) / rangeA) * (mTargetMax - mTargetMin)
                     + mTargetMin;
            }
         }
         else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
            for (int k = 0; k < numNeurons; k++) {
               int kExt = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.dn,
                     loc->halo.up);
               ABatch[kExt] = (float)0;
            }
         }
      }
      else if (mMethodCode == MEANSTD) {
         float sum   = 0;
         float sumsq = 0;
// Find sum of originalA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif
         for (int k = 0; k < numNeurons; k++) {
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  locOriginal->halo.rt,
                  locOriginal->halo.dn,
                  locOriginal->halo.up);
            sum += originalABatch[kextOriginal];
         }

         MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, mCommunicator->communicator());

         float mean = sum / numGlobalNeurons;

// Find (val - mean)^2 of originalA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sumsq)
#endif
         for (int k = 0; k < numNeurons; k++) {
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  locOriginal->halo.rt,
                  locOriginal->halo.dn,
                  locOriginal->halo.up);
            auto diffFromMean = originalABatch[kextOriginal] - mean;
            sumsq += diffFromMean * diffFromMean;
         }

         MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, mCommunicator->communicator());
         float stdev = std::sqrt(sumsq / numGlobalNeurons);
         // The difference between the if and the else clauses is only in the computation of
         // A[kext], but this way the stdev != 0.0 conditional is only evaluated once, not every
         // time through the for-loop.
         if (stdev != 0.0f) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int k = 0; k < numNeurons; k++) {
               int kext = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.up,
                     loc->halo.dn);
               int kextOriginal = kIndexExtended(
                     k,
                     locOriginal->nx,
                     locOriginal->ny,
                     locOriginal->nf,
                     locOriginal->halo.lt,
                     locOriginal->halo.rt,
                     locOriginal->halo.dn,
                     locOriginal->halo.up);
               ABatch[kext] =
                     ((originalABatch[kextOriginal] - mean) * (mTargetStd / stdev) + mTargetMean);
            }
         }
         else {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int k = 0; k < numNeurons; k++) {
               int kext = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.up,
                     loc->halo.dn);
               int kextOriginal = kIndexExtended(
                     k,
                     locOriginal->nx,
                     locOriginal->ny,
                     locOriginal->nf,
                     locOriginal->halo.lt,
                     locOriginal->halo.rt,
                     locOriginal->halo.dn,
                     locOriginal->halo.up);
               ABatch[kext] = originalABatch[kextOriginal];
            }
         }
      }
      else if (mMethodCode == L2) {
         float sum   = 0;
         float sumsq = 0;
// Find sum of originalA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sum)
#endif
         for (int k = 0; k < numNeurons; k++) {
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  locOriginal->halo.rt,
                  locOriginal->halo.dn,
                  locOriginal->halo.up);
            sum += originalABatch[kextOriginal];
         }

         MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, mCommunicator->communicator());

         float mean = sum / numGlobalNeurons;

// Find (val - mean)^2 of originalA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sumsq)
#endif
         for (int k = 0; k < numNeurons; k++) {
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  locOriginal->halo.rt,
                  locOriginal->halo.dn,
                  locOriginal->halo.up);
            sumsq += (originalABatch[kextOriginal] - mean) * (originalABatch[kextOriginal] - mean);
         }

         MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, mCommunicator->communicator());
         float stdev = std::sqrt(sumsq / numGlobalNeurons);
         // The difference between the if and the else clauses is only in the computation of
         // A[kext], but this way the stdev != 0.0 conditional is only evaluated once, not every
         // time through the for-loop.
         if (stdev != 0.0f) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int k = 0; k < numNeurons; k++) {
               int kext = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.up,
                     loc->halo.dn);
               int kextOriginal = kIndexExtended(
                     k,
                     locOriginal->nx,
                     locOriginal->ny,
                     locOriginal->nf,
                     locOriginal->halo.lt,
                     locOriginal->halo.rt,
                     locOriginal->halo.dn,
                     locOriginal->halo.up);
               ABatch[kext] =
                     ((originalABatch[kextOriginal] - mean)
                      * (1.0f / (stdev * std::sqrt((float)mPatchSize))));
            }
         }
         else {
            WarnLog() << "std of layer " << mOriginalBuffer->getName()
                      << " is 0, layer remains unchanged\n";
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int k = 0; k < numNeurons; k++) {
               int kext = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.up,
                     loc->halo.dn);
               int kextOriginal = kIndexExtended(
                     k,
                     locOriginal->nx,
                     locOriginal->ny,
                     locOriginal->nf,
                     locOriginal->halo.lt,
                     locOriginal->halo.rt,
                     locOriginal->halo.dn,
                     locOriginal->halo.up);
               ABatch[kext] = originalABatch[kextOriginal];
            }
         }
      }
      else if (mMethodCode == L2NOMEAN) {
         float sumsq = 0;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : sumsq)
#endif
         for (int k = 0; k < numNeurons; k++) {
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  locOriginal->halo.rt,
                  locOriginal->halo.dn,
                  locOriginal->halo.up);
            sumsq += (originalABatch[kextOriginal]) * (originalABatch[kextOriginal]);
         }

#ifdef PV_USE_MPI
         MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, mCommunicator->communicator());
#endif // PV_USE_MPI

         float stdev = std::sqrt(sumsq / numGlobalNeurons);
         // The difference between the if and the else clauses is only in the computation of
         // A[kext], but this way the stdev != 0.0 conditional is only evaluated once, not every
         // time through the for-loop.
         if (stdev != 0.0f) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int k = 0; k < numNeurons; k++) {
               int kext = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.up,
                     loc->halo.dn);
               int kextOriginal = kIndexExtended(
                     k,
                     locOriginal->nx,
                     locOriginal->ny,
                     locOriginal->nf,
                     locOriginal->halo.lt,
                     locOriginal->halo.rt,
                     locOriginal->halo.dn,
                     locOriginal->halo.up);
               ABatch[kext] =
                     ((originalABatch[kextOriginal]) * (1.0f / (stdev * std::sqrt((float)mPatchSize))));
            }
         }
         else {
            WarnLog() << "std of layer " << mOriginalBuffer->getName()
                      << " is 0, layer remains unchanged\n";
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int k = 0; k < numNeurons; k++) {
               int kext = kIndexExtended(
                     k,
                     loc->nx,
                     loc->ny,
                     loc->nf,
                     loc->halo.lt,
                     loc->halo.rt,
                     loc->halo.up,
                     loc->halo.dn);
               int kextOriginal = kIndexExtended(
                     k,
                     locOriginal->nx,
                     locOriginal->ny,
                     locOriginal->nf,
                     locOriginal->halo.lt,
                     locOriginal->halo.rt,
                     locOriginal->halo.dn,
                     locOriginal->halo.up);
               ABatch[kext] = originalABatch[kextOriginal];
            }
         }
      }
      else if (mMethodCode == POINTRESPONSENORMALIZATION) {
         int nx                 = loc->nx;
         int ny                 = loc->ny;
         int nf                 = loc->nf;
         PVHalo const *halo     = &loc->halo;
         PVHalo const *haloOrig = &locOriginal->halo;
// Loop through all nx and ny
// each y value specifies a different target so ok to thread here (sum, sumsq are defined inside
// loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int iY = 0; iY < ny; iY++) {
            for (int iX = 0; iX < nx; iX++) {
               // Find sum sq in feature space
               float sumsq = 0;
               for (int iF = 0; iF < nf; iF++) {
                  int kext = kIndex(
                        iX,
                        iY,
                        iF,
                        nx + haloOrig->lt + haloOrig->rt,
                        ny + haloOrig->dn + haloOrig->up,
                        nf);
                  sumsq += (originalABatch[kext]) * (originalABatch[kext]);
               }
               float divisor = std::sqrt(sumsq);
               // Difference in the if-part and else-part is only in the value assigned to A[kext],
               // but this way the divisor != 0 conditional does not have to be reevaluated every
               // time through the for loop. Can't pragma omp parallel the for loops because it was
               // already parallelized in the outermost for-loop
               if (divisor != 0) {
                  for (int iF = 0; iF < nf; iF++) {
                     int kextOrig = kIndex(
                           iX,
                           iY,
                           iF,
                           nx + haloOrig->lt + haloOrig->rt,
                           ny + haloOrig->dn + haloOrig->up,
                           nf);
                     int kext = kIndex(
                           iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf);
                     ABatch[kext] = (originalABatch[kextOrig] / divisor);
                  }
               }
               else {
                  for (int iF = 0; iF < nf; iF++) {
                     int kextOrig = kIndex(
                           iX,
                           iY,
                           iF,
                           nx + haloOrig->lt + haloOrig->rt,
                           ny + haloOrig->dn + haloOrig->up,
                           nf);
                     int kext = kIndex(
                           iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf);
                     ABatch[kext] = originalABatch[kextOrig];
                  }
               }
            }
         }
      }
      else if (mMethodCode == POINTMEANSTD) {
         int nx                 = loc->nx;
         int ny                 = loc->ny;
         int nf                 = loc->nf;
         PVHalo const *halo     = &loc->halo;
         PVHalo const *haloOrig = &locOriginal->halo;
// Loop through all nx and ny
// each y value specifies a different target so ok to thread here (sum, sumsq are defined inside
// loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int iY = 0; iY < ny; iY++) {
            for (int iX = 0; iX < nx; iX++) {
               // Find sum and sum sq in feature space
               float sum   = 0;
               float sumsq = 0;
               for (int iF = 0; iF < nf; iF++) {
                  int kext = kIndex(
                        iX,
                        iY,
                        iF,
                        nx + haloOrig->lt + haloOrig->rt,
                        ny + haloOrig->dn + haloOrig->up,
                        nf);
                  sum += originalABatch[kext];
               }
               float mean = sum / nf;
               for (int iF = 0; iF < nf; iF++) {
                  int kext = kIndex(
                        iX,
                        iY,
                        iF,
                        nx + haloOrig->lt + haloOrig->rt,
                        ny + haloOrig->dn + haloOrig->up,
                        nf);
                  sumsq += (originalABatch[kext] - mean) * (originalABatch[kext] - mean);
               }
               float stdev = std::sqrt(sumsq / nf);
               // Difference in the if-part and else-part is only in the value assigned to A[kext],
               // but this way the stdev != 0 conditional does not have to be reevaluated every
               // time through the for loop. Can't pragma omp parallel the for loops because it was
               // already parallelized in the outermost for-loop
               if (stdev != 0.0f) {
                  for (int iF = 0; iF < nf; iF++) {
                     int kextOrig = kIndex(
                           iX,
                           iY,
                           iF,
                           nx + haloOrig->lt + haloOrig->rt,
                           ny + haloOrig->dn + haloOrig->up,
                           nf);
                     int kext = kIndex(
                           iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf);
                     ABatch[kext] =
                           ((originalABatch[kextOrig] - mean) * (mTargetStd / stdev) + mTargetMean);
                  }
               }
               else {
                  for (int iF = 0; iF < nf; iF++) {
                     int kextOrig = kIndex(
                           iX,
                           iY,
                           iF,
                           nx + haloOrig->lt + haloOrig->rt,
                           ny + haloOrig->dn + haloOrig->up,
                           nf);
                     int kext = kIndex(
                           iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf);
                     ABatch[kext] = originalABatch[kextOrig];
                  }
               }
            }
         }
      }
      else if (mMethodCode == SOFTMAX) {
         int nx                 = loc->nx;
         int ny                 = loc->ny;
         int nf                 = loc->nf;
         PVHalo const *halo     = &loc->halo;
         PVHalo const *haloOrig = &locOriginal->halo;
// Loop through all nx and ny
// each y value specifies a different target so ok to thread here (sum, sumsq are defined inside
// loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int iY = 0; iY < ny; iY++) {
            for (int iX = 0; iX < nx; iX++) {
               float sumexpx = 0;
               // To prevent overflow, we subtract the max raw value before taking the exponential
               float maxvalue = -FLT_MAX;
               for (int iF = 0; iF < nf; iF++) {
                  int kextOrig = kIndex(
                        iX,
                        iY,
                        iF,
                        nx + haloOrig->lt + haloOrig->rt,
                        ny + haloOrig->dn + haloOrig->up,
                        nf);
                  maxvalue = std::max(maxvalue, originalABatch[kextOrig]);
               }
               for (int iF = 0; iF < nf; iF++) {
                  int kextOrig = kIndex(
                        iX,
                        iY,
                        iF,
                        nx + haloOrig->lt + haloOrig->rt,
                        ny + haloOrig->dn + haloOrig->up,
                        nf);
                  sumexpx += std::exp(originalABatch[kextOrig] - maxvalue);
               }
               // can't pragma omp parallel the for loops because it was already parallelized in the
               // outermost for-loop
               for (int iF = 0; iF < nf; iF++) {
                  int kextOrig = kIndex(
                        iX,
                        iY,
                        iF,
                        nx + haloOrig->lt + haloOrig->rt,
                        ny + haloOrig->dn + haloOrig->up,
                        nf);
                  int kext =
                        kIndex(iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf);
                  if (sumexpx != 0.0f && sumexpx == sumexpx) { // Check for zero and NaN
                     ABatch[kext] = std::exp(originalABatch[kextOrig] - maxvalue) / sumexpx;
                  }
                  else {
                     ABatch[kext] = 0.0f;
                  }
                  pvAssert(ABatch[kext] >= 0 && ABatch[kext] <= 1);
               }
            }
         }
      }
      else if (mMethodCode == LOGREG) {
// Loop through all nx and ny
// each y value specifies a different target so ok to thread here (sum, sumsq are defined inside
// loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int k = 0; k < numNeurons; k++) {
            int kext = kIndexExtended(
                  k,
                  loc->nx,
                  loc->ny,
                  loc->nf,
                  loc->halo.lt,
                  loc->halo.rt,
                  loc->halo.up,
                  loc->halo.dn);
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  locOriginal->halo.lt,
                  locOriginal->halo.rt,
                  locOriginal->halo.dn,
                  locOriginal->halo.up);
            ABatch[kext] = 1.0f / (1.0f + std::exp(originalABatch[kextOriginal]));
         }
      }
      else if (mMethodCode == ZEROTONEGATIVE) {
         PVHalo const *halo     = &loc->halo;
         PVHalo const *haloOrig = &locOriginal->halo;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int k = 0; k < numNeurons; k++) {
            int kextOriginal = kIndexExtended(
                  k,
                  locOriginal->nx,
                  locOriginal->ny,
                  locOriginal->nf,
                  haloOrig->lt,
                  haloOrig->rt,
                  haloOrig->dn,
                  haloOrig->up);
            int kext = kIndexExtended(
                  k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
            if (originalABatch[kextOriginal] == 0) {
               ;
               ABatch[kext] = -1;
            }
            else {
               ABatch[kext] = originalABatch[kextOriginal];
            }
         }
      }
   }
}

} // end namespace PV
