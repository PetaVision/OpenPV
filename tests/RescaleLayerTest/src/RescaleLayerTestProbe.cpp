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
#include <string.h>
#include <utils/PVLog.hpp>

namespace PV {

RescaleLayerTestProbe::RescaleLayerTestProbe(const char *name, HyPerCol *hc) : StatsProbe() {
   initialize(name, hc);
}

int RescaleLayerTestProbe::initialize_base() { return PV_SUCCESS; }

int RescaleLayerTestProbe::initialize(const char *name, HyPerCol *hc) {
   return StatsProbe::initialize(name, hc);
}

void RescaleLayerTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufActivity); }

int RescaleLayerTestProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = StatsProbe::communicateInitInfo(message);
   FatalIf(!(getTargetLayer()), "Test failed.\n");
   RescaleLayer *targetRescaleLayer = dynamic_cast<RescaleLayer *>(getTargetLayer());
   if (targetRescaleLayer == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "RescaleLayerTestProbe: targetLayer \"%s\" is not a RescaleLayer.\n",
               this->getTargetName());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

int RescaleLayerTestProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   if (timed == parent->getStartTime()) {
      return PV_SUCCESS;
   }
   float tolerance      = 2.0e-5f;
   Communicator *icComm = parent->getCommunicator();
   bool isRoot          = icComm->commRank() == 0;

   RescaleLayer *targetRescaleLayer = dynamic_cast<RescaleLayer *>(getTargetLayer());
   FatalIf(!(targetRescaleLayer), "Test failed.\n");

   if (targetRescaleLayer->getRescaleMethod() == NULL) {
      ErrorLog().printf(
            "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" does not have rescaleMethod set.  "
            "Exiting.\n",
            name,
            targetRescaleLayer->getName());
      status = PV_FAILURE;
   }
   else if (!strcmp(targetRescaleLayer->getRescaleMethod(), "maxmin")) {
      if (!isRoot) {
         return PV_SUCCESS;
      }
      for (int b = 0; b < parent->getNBatch(); b++) {
         float targetMax = targetRescaleLayer->getTargetMax();
         if (fabsf(fMax[b] - targetMax) > tolerance) {
            ErrorLog().printf(
                  "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has max %f instead of target "
                  "max %f\n",
                  getName(),
                  targetRescaleLayer->getName(),
                  (double)fMax[b],
                  (double)targetMax);
            status = PV_FAILURE;
         }
         float targetMin = targetRescaleLayer->getTargetMin();
         if (fabsf(fMin[b] - targetMin) > tolerance) {
            ErrorLog().printf(
                  "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has min %f instead of target "
                  "min %f\n",
                  getName(),
                  targetRescaleLayer->getName(),
                  (double)fMin[b],
                  (double)targetMin);
            status = PV_FAILURE;
         }

         // Now, check whether rescaled activity and original V are colinear.
         PVLayerLoc const *rescaleLoc = targetRescaleLayer->getLayerLoc();
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
         float const *rescaledData = targetRescaleLayer->getLayerData()
                                     + b * targetRescaleLayer->getNumExtended()
                                     + rescaleExtendedOffset;
         PVLayerLoc const *origLoc = targetRescaleLayer->getOriginalLayer()->getLayerLoc();
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
         float const *origData = targetRescaleLayer->getOriginalLayer()->getLayerData()
                                 + b * targetRescaleLayer->getOriginalLayer()->getNumExtended()
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
                  targetRescaleLayer->getDescription_c());
            status = PV_FAILURE;
         }
      }
   }
   // l2 norm with a patch size of 1 (default) should be the same as rescaling with meanstd with
   // target mean 0 and std of 1/sqrt(patchsize)
   else if (
         !strcmp(targetRescaleLayer->getRescaleMethod(), "meanstd")
         || !strcmp(targetRescaleLayer->getRescaleMethod(), "l2")) {
      if (!isRoot) {
         return PV_SUCCESS;
      }
      for (int b = 0; b < parent->getNBatch(); b++) {
         float targetMean, targetStd;
         if (!strcmp(targetRescaleLayer->getRescaleMethod(), "meanstd")) {
            targetMean = targetRescaleLayer->getTargetMean();
            targetStd  = targetRescaleLayer->getTargetStd();
         }
         else {
            targetMean = 0;
            targetStd  = 1 / sqrtf((float)targetRescaleLayer->getL2PatchSize());
         }

         if (fabsf(avg[b] - targetMean) > tolerance) {
            ErrorLog().printf(
                  "%s: %s has mean %f instead of target mean %f\n",
                  getDescription_c(),
                  targetRescaleLayer->getDescription_c(),
                  (double)avg[b],
                  (double)targetMean);
            status = PV_FAILURE;
         }
         if (sigma[b] > tolerance && fabsf(sigma[b] - targetStd) > tolerance) {
            ErrorLog().printf(
                  "%s: %s has std.dev. %f instead of target std.dev. %f\n",
                  getDescription_c(),
                  targetRescaleLayer->getDescription_c(),
                  (double)sigma[b],
                  (double)targetStd);
            status = PV_FAILURE;
         }

         // Now, check whether rescaled activity and original V are colinear.
         PVLayerLoc const *rescaleLoc = targetRescaleLayer->getLayerLoc();
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
         float const *rescaledData = targetRescaleLayer->getLayerData()
                                     + b * targetRescaleLayer->getNumExtended()
                                     + rescaleExtendedOffset;
         PVLayerLoc const *origLoc = targetRescaleLayer->getOriginalLayer()->getLayerLoc();
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
         float const *origData = targetRescaleLayer->getOriginalLayer()->getLayerData()
                                 + b * targetRescaleLayer->getOriginalLayer()->getNumExtended()
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
                  targetRescaleLayer->getDescription_c());
            status = PV_FAILURE;
         }
      }
   }
   else if (!strcmp(targetRescaleLayer->getRescaleMethod(), "pointmeanstd")) {
      PVLayerLoc const *loc = targetRescaleLayer->getLayerLoc();
      int nf                = loc->nf;
      if (nf < 2) {
         return PV_SUCCESS;
      }
      PVHalo const *halo = &loc->halo;
      float targetMean   = targetRescaleLayer->getTargetMean();
      float targetStd    = targetRescaleLayer->getTargetStd();
      int numNeurons     = targetRescaleLayer->getNumNeurons();
      for (int b = 0; b < parent->getNBatch(); b++) {
         float const *originalData =
               targetRescaleLayer->getV() + b * targetRescaleLayer->getNumNeurons();
         float const *rescaledData =
               targetRescaleLayer->getLayerData() + b * targetRescaleLayer->getNumExtended();
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
                     targetRescaleLayer->getName(),
                     parent->columnId(),
                     k,
                     (double)pointmean,
                     (double)targetMean);
               status = PV_FAILURE;
            }
            if (pointstd > tolerance && fabsf(pointstd - targetStd) > tolerance) {
               ErrorLog().printf(
                     "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", location in rank %d, "
                     "starting at restricted neuron %d, has std.dev. %f instead of target std.dev. "
                     "%f\n",
                     getName(),
                     targetRescaleLayer->getName(),
                     parent->columnId(),
                     k,
                     (double)pointstd,
                     (double)targetStd);
               status = PV_FAILURE;
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
                     targetRescaleLayer->getName(),
                     parent->columnId(),
                     k);
               status = PV_FAILURE;
            }
         }
      }
   }
   else if (!strcmp(targetRescaleLayer->getRescaleMethod(), "zerotonegative")) {
      int numNeurons = targetRescaleLayer->getNumNeurons();
      FatalIf(
            !(numNeurons == targetRescaleLayer->getOriginalLayer()->getNumNeurons()),
            "Test failed.\n");
      PVLayerLoc const *rescaleLoc = targetRescaleLayer->getLayerLoc();
      PVHalo const *rescaleHalo    = &rescaleLoc->halo;
      int nf                       = rescaleLoc->nf;
      HyPerLayer *originalLayer    = targetRescaleLayer->getOriginalLayer();
      PVLayerLoc const *origLoc    = originalLayer->getLayerLoc();
      PVHalo const *origHalo       = &origLoc->halo;
      FatalIf(!(origLoc->nf == nf), "Test failed.\n");

      for (int b = 0; b < parent->getNBatch(); b++) {
         float const *rescaledData =
               targetRescaleLayer->getLayerData() + b * targetRescaleLayer->getNumExtended();
         float const *originalData =
               originalLayer->getLayerData() + b * originalLayer->getNumExtended();
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
                     targetRescaleLayer->getName(),
                     parent->columnId(),
                     k,
                     (double)observedval,
                     (double)correctval);
               status = PV_FAILURE;
            }
         }
      }
   }
   else {
      FatalIf(!(0), "Test failed.\n"); // All allowable rescaleMethod values are handled above.
   }
   if (status == PV_FAILURE) {
      exit(EXIT_FAILURE);
   }
   return status;
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
