/*
 * RescaleLayerTestProbe.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#include "RescaleLayerTestProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <layers/RescaleLayer.hpp>
#include <utils/PVLog.hpp>

#include <string.h>

namespace PV {

RescaleLayerTestProbe::RescaleLayerTestProbe(const char *name, PVParams *params, Communicator *comm)
      : StatsProbe() {
   initialize(name, params, comm);
}

void RescaleLayerTestProbe::initialize(const char *name, PVParams *params, Communicator *comm) {
   StatsProbe::initialize(name, params, comm);
}

void RescaleLayerTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

Response::Status RescaleLayerTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = StatsProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   RescaleLayer *targetRescaleLayer = dynamic_cast<RescaleLayer *>(getTargetLayer());
   if (targetRescaleLayer == nullptr) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "RescaleLayerTestProbe: targetLayer \"%s\" is not a RescaleLayer.\n",
               this->getTargetName());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   auto *targetActivityComponent = targetRescaleLayer->getComponentByType<ActivityComponent>();
   if (targetActivityComponent == nullptr) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "RescaleLayerTestProbe: targetLayer \"%s\" does not have an ActivityComponent.\n",
               this->getTargetName());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }

   mRescaleBuffer = targetActivityComponent->getComponentByType<RescaleActivityBuffer>();
   if (mRescaleBuffer == nullptr) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "RescaleLayerTestProbe: targetLayer \"%s\" does not have a RescaleActivityBuffer.\n",
               this->getTargetName());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

Response::Status RescaleLayerTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (simTime == 0.0) {
      return status;
   }
   float tolerance      = 2.0e-5f;
   Communicator *icComm = mCommunicator;
   bool isRoot          = icComm->commRank() == 0;

   pvAssert(mRescaleBuffer);

   bool failed = false;
   if (mRescaleBuffer->getRescaleMethod() == NULL) {
      ErrorLog().printf(
            "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" does not have rescaleMethod set.  "
            "Exiting.\n",
            name,
            mRescaleBuffer->getName());
      failed = true;
   }
   else if (!strcmp(mRescaleBuffer->getRescaleMethod(), "maxmin")) {
      if (!isRoot) {
         return Response::SUCCESS;
      }
      for (int b = 0; b < mLocalBatchWidth; b++) {
         float targetMax = mRescaleBuffer->getTargetMax();
         if (fabsf(fMax[b] - targetMax) > tolerance) {
            ErrorLog().printf(
                  "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has max %f instead of target "
                  "max %f\n",
                  getName(),
                  mRescaleBuffer->getName(),
                  (double)fMax[b],
                  (double)targetMax);
            failed = true;
         }
         float targetMin = mRescaleBuffer->getTargetMin();
         if (fabsf(fMin[b] - targetMin) > tolerance) {
            ErrorLog().printf(
                  "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has min %f instead of target "
                  "min %f\n",
                  getName(),
                  mRescaleBuffer->getName(),
                  (double)fMin[b],
                  (double)targetMin);
            failed = true;
         }

         // Now, check whether rescaled activity and original V are colinear.
         PVLayerLoc const *rescaleLoc = mRescaleBuffer->getLayerLoc();
         PVHalo const *rescaleHalo    = &rescaleLoc->halo;
         int nk                       = rescaleLoc->nx * rescaleLoc->nf;
         int ny                       = rescaleLoc->ny;
         int rescaleStrideYExtended =
               (rescaleLoc->nx + rescaleHalo->lt + rescaleHalo->rt) * rescaleLoc->nf;
         int rescaleExtendedOffset = kIndexExtended(
               0,
               rescaleLoc->nx,
               rescaleLoc->ny,
               rescaleLoc->nf,
               rescaleHalo->lt,
               rescaleHalo->rt,
               rescaleHalo->dn,
               rescaleHalo->up);
         float const *rescaledData = mRescaleBuffer->getBufferData()
                                     + b * mRescaleBuffer->getBufferSize() + rescaleExtendedOffset;
         PVLayerLoc const *origLoc = mRescaleBuffer->getOriginalBuffer()->getLayerLoc();
         PVHalo const *origHalo    = &origLoc->halo;
         FatalIf(!(nk == origLoc->nx * origLoc->nf), "Test failed.\n");
         FatalIf(!(ny == origLoc->ny), "Test failed.\n");
         int origStrideYExtended = (origLoc->nx + origHalo->lt + origHalo->rt) * origLoc->nf;
         int origExtendedOffset  = kIndexExtended(
               0,
               rescaleLoc->nx,
               rescaleLoc->ny,
               rescaleLoc->nf,
               rescaleHalo->lt,
               rescaleHalo->rt,
               rescaleHalo->dn,
               rescaleHalo->up);
         float const *origData = mRescaleBuffer->getOriginalBuffer()->getBufferData()
                                 + b * mRescaleBuffer->getOriginalBuffer()->getBufferSize()
                                 + origExtendedOffset;

         bool iscolinear = colinear(
               nk,
               ny,
               origStrideYExtended,
               rescaleStrideYExtended,
               origData,
               rescaledData,
               tolerance,
               NULL,
               NULL,
               NULL);
         if (!iscolinear) {
            ErrorLog().printf(
                  "%s: %s data is not a linear rescaling of original membrane potential.\n",
                  getDescription_c(),
                  mRescaleBuffer->getDescription_c());
            failed = true;
         }
      }
   }
   // l2 norm with a patch size of 1 (default) should be the same as rescaling with meanstd with
   // target mean 0 and std of 1/sqrt(patchsize)
   else if (
         !strcmp(mRescaleBuffer->getRescaleMethod(), "meanstd")
         || !strcmp(mRescaleBuffer->getRescaleMethod(), "l2")) {
      if (!isRoot) {
         return Response::SUCCESS;
      }
      for (int b = 0; b < mLocalBatchWidth; b++) {
         float targetMean, targetStd;
         if (!strcmp(mRescaleBuffer->getRescaleMethod(), "meanstd")) {
            targetMean = mRescaleBuffer->getTargetMean();
            targetStd  = mRescaleBuffer->getTargetStd();
         }
         else {
            targetMean = 0;
            targetStd  = 1 / sqrtf((float)mRescaleBuffer->getPatchSize());
         }

         if (fabsf(avg[b] - targetMean) > tolerance) {
            ErrorLog().printf(
                  "%s: %s has mean %f instead of target mean %f\n",
                  getDescription_c(),
                  mRescaleBuffer->getDescription_c(),
                  (double)avg[b],
                  (double)targetMean);
            failed = true;
         }
         if (sigma[b] > tolerance && fabsf(sigma[b] - targetStd) > tolerance) {
            ErrorLog().printf(
                  "%s: %s has std.dev. %f instead of target std.dev. %f\n",
                  getDescription_c(),
                  mRescaleBuffer->getDescription_c(),
                  (double)sigma[b],
                  (double)targetStd);
            failed = true;
         }

         // Now, check whether rescaled activity and original V are colinear.
         PVLayerLoc const *rescaleLoc = mRescaleBuffer->getLayerLoc();
         PVHalo const *rescaleHalo    = &rescaleLoc->halo;
         int nk                       = rescaleLoc->nx * rescaleLoc->nf;
         int ny                       = rescaleLoc->ny;
         int rescaleStrideYExtended =
               (rescaleLoc->nx + rescaleHalo->lt + rescaleHalo->rt) * rescaleLoc->nf;
         int rescaleExtendedOffset = kIndexExtended(
               0,
               rescaleLoc->nx,
               rescaleLoc->ny,
               rescaleLoc->nf,
               rescaleHalo->lt,
               rescaleHalo->rt,
               rescaleHalo->dn,
               rescaleHalo->up);
         float const *rescaledData = mRescaleBuffer->getBufferData() // should be getLayerData()
                                     + b * mRescaleBuffer->getBufferSize() + rescaleExtendedOffset;
         PVLayerLoc const *origLoc = mRescaleBuffer->getOriginalBuffer()->getLayerLoc();
         PVHalo const *origHalo    = &origLoc->halo;
         FatalIf(!(nk == origLoc->nx * origLoc->nf), "Test failed.\n");
         FatalIf(!(ny == origLoc->ny), "Test failed.\n");
         int origStrideYExtended = (origLoc->nx + origHalo->lt + origHalo->rt) * origLoc->nf;
         int origExtendedOffset  = kIndexExtended(
               0,
               rescaleLoc->nx,
               rescaleLoc->ny,
               rescaleLoc->nf,
               rescaleHalo->lt,
               rescaleHalo->rt,
               rescaleHalo->dn,
               rescaleHalo->up);
         float const *origData = mRescaleBuffer->getOriginalBuffer()->getBufferData()
                                 + b * mRescaleBuffer->getOriginalBuffer()->getBufferSize()
                                 + origExtendedOffset;

         bool iscolinear = colinear(
               nk,
               ny,
               origStrideYExtended,
               rescaleStrideYExtended,
               origData,
               rescaledData,
               tolerance,
               NULL,
               NULL,
               NULL);
         if (!iscolinear) {
            ErrorLog().printf(
                  "%s: %s data is not a linear rescaling of original membrane potential.\n",
                  getDescription_c(),
                  mRescaleBuffer->getDescription_c());
            failed = true;
         }
      }
   }
   else if (!strcmp(mRescaleBuffer->getRescaleMethod(), "pointmeanstd")) {
      PVLayerLoc const *loc = mRescaleBuffer->getLayerLoc();
      int nf                = loc->nf;
      if (nf < 2) {
         return Response::SUCCESS;
      }
      PVHalo const *halo = &loc->halo;
      float targetMean   = mRescaleBuffer->getTargetMean();
      float targetStd    = mRescaleBuffer->getTargetStd();
      int numNeurons     = loc->nx * loc->ny * loc->nf;
      for (int b = 0; b < mLocalBatchWidth; b++) {
         float const *originalData = mRescaleBuffer->getOriginalBuffer()->getBufferData(b);
         float const *rescaledData = mRescaleBuffer->getBufferData(b);
         for (int k = 0; k < numNeurons; k += nf) {
            int kExtended = kIndexExtended(
                  k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
            float pointmean = 0.0f;
            for (int f = 0; f < nf; f++) {
               pointmean += rescaledData[kExtended + f];
            }
            pointmean /= nf;
            float pointstd = 0.0f;
            for (int f = 0; f < nf; f++) {
               float d = rescaledData[kExtended + f] - pointmean;
               pointstd += d * d;
            }
            pointstd /= nf;
            pointstd = sqrtf(pointstd);
            if (fabsf(pointmean - targetMean) > tolerance) {
               ErrorLog().printf(
                     "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", location in rank %d, "
                     "starting at restricted neuron %d, has mean %f instead of target mean %f\n",
                     getName(),
                     mRescaleBuffer->getName(),
                     mCommunicator->globalCommRank(),
                     k,
                     (double)pointmean,
                     (double)targetMean);
               failed = true;
            }
            if (pointstd > tolerance && fabsf(pointstd - targetStd) > tolerance) {
               ErrorLog().printf(
                     "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", location in rank %d, "
                     "starting at restricted neuron %d, has std.dev. %f instead of target std.dev. "
                     "%f\n",
                     getName(),
                     mRescaleBuffer->getName(),
                     mCommunicator->globalCommRank(),
                     k,
                     (double)pointstd,
                     (double)targetStd);
               failed = true;
            }
            bool iscolinear = colinear(
                  nf,
                  1,
                  0,
                  0,
                  &originalData[k],
                  &rescaledData[kExtended],
                  tolerance,
                  NULL,
                  NULL,
                  NULL);
            if (!iscolinear) {
               ErrorLog().printf(
                     "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", location in rank %d, "
                     "starting at restricted neuron %d, is not a linear rescaling.\n",
                     getName(),
                     mRescaleBuffer->getName(),
                     mCommunicator->globalCommRank(),
                     k);
               failed = true;
            }
         }
      }
   }
   else if (!strcmp(mRescaleBuffer->getRescaleMethod(), "zerotonegative")) {
      PVLayerLoc const *rescaleLoc = mRescaleBuffer->getLayerLoc();
      PVHalo const *rescaleHalo    = &rescaleLoc->halo;
      int nf                       = rescaleLoc->nf;
      auto const *originalBuffer   = mRescaleBuffer->getOriginalBuffer();
      PVLayerLoc const *origLoc    = originalBuffer->getLayerLoc();
      PVHalo const *origHalo       = &origLoc->halo;
      FatalIf(!(origLoc->nf == nf), "Test failed.\n");
      int const numNeurons = rescaleLoc->nx * rescaleLoc->ny * rescaleLoc->nf;

      for (int b = 0; b < mLocalBatchWidth; b++) {
         float const *rescaledData = mRescaleBuffer->getBufferData(b);
         float const *originalData = originalBuffer->getBufferData(b);
         for (int k = 0; k < numNeurons; k++) {
            int rescale_kExtended = kIndexExtended(
                  k,
                  rescaleLoc->nx,
                  rescaleLoc->ny,
                  rescaleLoc->nf,
                  rescaleHalo->lt,
                  rescaleHalo->rt,
                  rescaleHalo->dn,
                  rescaleHalo->up);
            int orig_kExtended = kIndexExtended(
                  k,
                  origLoc->nx,
                  origLoc->ny,
                  origLoc->nf,
                  origHalo->lt,
                  origHalo->rt,
                  origHalo->dn,
                  origHalo->up);
            float observedval = rescaledData[rescale_kExtended];
            float correctval  = originalData[orig_kExtended] ? observedval : -1.0f;
            if (observedval != correctval) {
               ErrorLog().printf(
                     "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", rank %d, restricted "
                     "neuron %d has value %f instead of expected %f\n.",
                     this->getName(),
                     mRescaleBuffer->getName(),
                     mCommunicator->globalCommRank(),
                     k,
                     (double)observedval,
                     (double)correctval);
               failed = true;
            }
         }
      }
   }
   else {
      Fatal().printf("Unrecognized rescaleMethod.\n");
   }
   if (failed) {
      exit(EXIT_FAILURE);
   }
   return Response::SUCCESS;
}

bool RescaleLayerTestProbe::colinear(
      int nx,
      int ny,
      int ystrideA,
      int ystrideB,
      float const *A,
      float const *B,
      double tolerance,
      double *cov,
      double *stdA,
      double *stdB) {
   int numelements = nx * ny;
   if (numelements <= 1) {
      return false;
   } // Need two or more points to be meaningful
   double amean = 0.0;
   double bmean = 0.0;
   for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
         amean += (double)A[x + ystrideA * y];
         bmean += (double)B[x + ystrideB * y];
      }
   }
   amean /= numelements;
   bmean /= numelements;

   double astd       = 0.0;
   double bstd       = 0.0;
   double covariance = 0.0;
   for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
         double d1 = ((double)A[x + ystrideA * y] - amean);
         astd += d1 * d1;
         double d2 = ((double)B[x + ystrideB * y] - bmean);
         bstd += d2 * d2;
         covariance += d1 * d2;
      }
   }
   astd /= numelements - 1;
   bstd /= numelements - 1;
   covariance /= numelements - 1;
   astd = sqrt(astd);
   bstd = sqrt(bstd);
   if (cov) {
      *cov = covariance;
   }
   if (stdA) {
      *stdA = astd;
   }
   if (stdB) {
      *stdB = bstd;
   }
   return fabs(covariance - astd * bstd) <= tolerance;
}

} /* namespace PV */
