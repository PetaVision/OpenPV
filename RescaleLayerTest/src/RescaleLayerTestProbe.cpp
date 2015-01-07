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
#include <assert.h>

namespace PV {

RescaleLayerTestProbe::RescaleLayerTestProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initRescaleLayerTestProbe(probeName, hc);
}

int RescaleLayerTestProbe::initRescaleLayerTestProbe_base() { return PV_SUCCESS; }

int RescaleLayerTestProbe::initRescaleLayerTestProbe(const char * probeName, HyPerCol * hc)
{
   return initStatsProbe(probeName, hc);
}

void RescaleLayerTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

int RescaleLayerTestProbe::communicateInitInfo() {
   int status = StatsProbe::communicateInitInfo();
   assert(getTargetLayer());
   RescaleLayer * targetRescaleLayer = dynamic_cast<RescaleLayer *>(getTargetLayer());
   if (targetRescaleLayer==NULL) {
      if (getParent()->columnId()==0) {
         fprintf(stderr, "RescaleLayerTestProbe Error: targetLayer \"%s\" is not a RescaleLayer.\n", this->getTargetName());
      }
      MPI_Barrier(getParent()->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

int RescaleLayerTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
   if (timed==getParent()->getStartTime()) { return PV_SUCCESS; }
   float tolerance = 1.0e-5f;
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   bool isRoot = icComm->commRank() == 0;
#else
   bool isRoot = true;
#endif // PV_USE_MPI

   RescaleLayer * targetRescaleLayer = dynamic_cast<RescaleLayer *>(getTargetLayer());
   assert(targetRescaleLayer);
   if (targetRescaleLayer->getRescaleMethod()==NULL) {
      fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" does not have rescaleMethod set.  Exiting.\n");
      status = PV_FAILURE;
   }
   else if (!strcmp(targetRescaleLayer->getRescaleMethod(), "maxmin")) {
      if (!isRoot) { return PV_SUCCESS; }
      float targetMax = targetRescaleLayer->getTargetMax();
      if (fabs(fMax-targetMax)>tolerance) {
         fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has max %f instead of target max %f\n", getName(), targetRescaleLayer->getName(), fMax, targetMax);
         status = PV_FAILURE;
      }
      float targetMin = targetRescaleLayer->getTargetMin();
      if (fabs(fMin-targetMin)>tolerance) {
         fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has min %f instead of target min %f\n", getName(), targetRescaleLayer->getName(), fMin, targetMin);
         status = PV_FAILURE;
      }
      // Now, check whether rescaled layer data and original layer data are colinear.
      // TODO: it would be better to use the rescaled layer's V instead of the original layer's A for comparison.
      // The rescaled layer's V won't have a halo, and it doesn't assume that original layer's transfer function is the identity.
      PVLayerLoc const * rescaledLoc = targetRescaleLayer->getLayerLoc();
      PVHalo const * rescaledHalo = &rescaledLoc->halo;
      PVLayerLoc const * originalLoc = targetRescaleLayer->getOriginalLayer()->getLayerLoc();
      PVHalo const * originalHalo = &originalLoc->halo;
      int nk = rescaledLoc->nx * rescaledLoc->nf;
      int ny = rescaledLoc->ny;
      assert(nk == originalLoc->nx * originalLoc->nf && ny == originalLoc->ny);
      int originalStrideY = originalLoc->nx + originalHalo->lt + originalHalo->rt;
      int rescaledStrideY = rescaledLoc->nx + rescaledHalo->lt + rescaledHalo->rt;
      int originalOffset = kIndexExtended(0, originalLoc->nx, originalLoc->ny, originalLoc->nf, originalHalo->lt, originalHalo->rt, originalHalo->dn, originalHalo->up);
      int rescaledOffset = kIndexExtended(0, rescaledLoc->nx, rescaledLoc->ny, rescaledLoc->nf, rescaledHalo->lt, rescaledHalo->rt, rescaledHalo->dn, rescaledHalo->up);
      pvadata_t const * originalData = targetRescaleLayer->getOriginalLayer()->getLayerData();
      pvadata_t const * rescaledData = targetRescaleLayer->getLayerData();
      bool iscolinear = colinear(nk, ny, originalStrideY, rescaledStrideY, &originalData[originalOffset], &rescaledData[rescaledOffset], tolerance, NULL, NULL, NULL);

      if (!iscolinear) {
         fprintf(stderr, "RescaleLayerTestProbe \"%s\": Rescale layer \"%s\" data is not a linear rescaling of original layer data.\n");
         status = PV_FAILURE;
      }
   }
   else if (!strcmp(targetRescaleLayer->getRescaleMethod(), "meanstd")) {
      if (!isRoot) { return PV_SUCCESS; }
      float targetMean = targetRescaleLayer->getTargetMean();
      if (fabs(avg-targetMean)>tolerance) {
         fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has mean %f instead of target mean %f\n", getName(), targetRescaleLayer->getName(), avg, targetMean);
         status = PV_FAILURE;
      }
      float targetStd = targetRescaleLayer->getTargetStd();
      if (sigma>tolerance && fabs(sigma-targetStd)>tolerance) {
         fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\" has std.dev. %f instead of target std.dev. %f\n", getName(), targetRescaleLayer->getName(), sigma, targetStd);
         status = PV_FAILURE;
      }
      
      // Now, check whether rescaled layer data and original layer data are colinear.
      PVLayerLoc const * rescaledLoc = targetRescaleLayer->getLayerLoc();
      PVHalo const * rescaledHalo = &rescaledLoc->halo;
      PVLayerLoc const * originalLoc = targetRescaleLayer->getOriginalLayer()->getLayerLoc();
      PVHalo const * originalHalo = &originalLoc->halo;
      int nk = rescaledLoc->nx * rescaledLoc->nf;
      int ny = rescaledLoc->ny;
      assert(nk == originalLoc->nx * originalLoc->nf && ny == originalLoc->ny);
      int originalStrideY = originalLoc->nx + originalHalo->lt + originalHalo->rt;
      int rescaledStrideY = rescaledLoc->nx + rescaledHalo->lt + rescaledHalo->rt;
      int originalOffset = kIndexExtended(0, originalLoc->nx, originalLoc->ny, originalLoc->nf, originalHalo->lt, originalHalo->rt, originalHalo->dn, originalHalo->up);
      int rescaledOffset = kIndexExtended(0, rescaledLoc->nx, rescaledLoc->ny, rescaledLoc->nf, rescaledHalo->lt, rescaledHalo->rt, rescaledHalo->dn, rescaledHalo->up);
      pvadata_t const * originalData = targetRescaleLayer->getOriginalLayer()->getLayerData();
      pvadata_t const * rescaledData = targetRescaleLayer->getLayerData();
      bool iscolinear = colinear(nk, ny, originalStrideY, rescaledStrideY, &originalData[originalOffset], &rescaledData[rescaledOffset], tolerance, NULL, NULL, NULL);

      if (!iscolinear) {
         fprintf(stderr, "RescaleLayerTestProbe \"%s\": Rescale layer \"%s\" data is not a linear rescaling of original layer data.\n");
         status = PV_FAILURE;
      }
   }
   else if (!strcmp(targetRescaleLayer->getRescaleMethod(), "pointmeanstd")) {
      PVLayerLoc const * loc = targetRescaleLayer->getLayerLoc();
      int nf = loc->nf;
      if (nf<2) { return PV_SUCCESS; }
      PVHalo const * halo = &loc->halo;
      HyPerLayer * originalLayer = targetRescaleLayer->getOriginalLayer();
      PVLayerLoc const * originalLoc = originalLayer->getLayerLoc();
      PVHalo const * originalHalo = &originalLoc->halo;
      pvadata_t const * rescaleddata = targetRescaleLayer->getLayerData();
      pvadata_t const * originaldata = originalLayer->getLayerData();
      float targetMean = targetRescaleLayer->getTargetMean();
      float targetStd = targetRescaleLayer->getTargetStd();
      int numNeurons = targetRescaleLayer->getNumNeurons();
      for (int k=0; k<numNeurons; k+=nf) {
         int kExtended = kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
         int kOriginalExtended = kIndexExtended(k, originalLoc->nx, originalLoc->ny, originalLoc->nf, originalHalo->lt, originalHalo->rt, originalHalo->dn, originalHalo->up);
         double pointmean = 0.0;
         for (int f=0; f<nf; f++) {
            pointmean += rescaleddata[kOriginalExtended+f];
         }
         pointmean /= nf;
         double pointstd = 0.0;
         for (int f=0; f<nf; f++) {
            double d = rescaleddata[kOriginalExtended+f]-pointmean;
            pointstd += d*d;
         }
         pointstd /= nf;
         pointstd = sqrt(pointstd);
         if (fabs(pointmean-targetMean)>tolerance) {
            fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", location in rank %d, starting at restricted neuron %d, has mean %f instead of target mean %f\n",
                  getName(), targetRescaleLayer->getName(), getParent()->columnId(), k, pointmean, targetMean);
            status = PV_FAILURE;
         }
         if (pointstd>tolerance && fabs(pointstd-targetStd)>tolerance) {
            fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", location in rank %d, starting at restricted neuron %d, has std.dev. %f instead of target std.dev. %f\n",
                  getName(), targetRescaleLayer->getName(), getParent()->columnId(), k, pointstd, targetStd);
            status = PV_FAILURE;
         }
         bool iscolinear = colinear(nf, 1, 0, 0, &originaldata[kOriginalExtended], &rescaleddata[kOriginalExtended], tolerance, NULL, NULL, NULL);
         if (!iscolinear) {
            fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", location in rank %d, starting at restricted neuron %d, is not colinear.\n");
            status = PV_FAILURE;
         }
      }
   }
   else if (!strcmp(targetRescaleLayer->getRescaleMethod(), "zerotonegative")) {
      int numNeurons = targetRescaleLayer->getNumNeurons();
      PVLayerLoc const * loc = targetRescaleLayer->getLayerLoc();
      PVHalo const * halo = &loc->halo;
      int nf = loc->nf;
      HyPerLayer * originalLayer = targetRescaleLayer->getOriginalLayer();
      PVLayerLoc const * originalLoc = originalLayer->getLayerLoc();
      PVHalo const * originalHalo = &originalLoc->halo;
      pvadata_t const * rescaleddata = targetRescaleLayer->getLayerData();
      pvadata_t const * originaldata = originalLayer->getLayerData();
      for (int k=0; k<numNeurons; k++) {
         int kExtended = kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
         int kOriginalExtended = kIndexExtended(k, originalLoc->nx, originalLoc->ny, originalLoc->nf, originalHalo->lt, originalHalo->rt, originalHalo->dn, originalHalo->up);
         pvadata_t correctval = originaldata[kOriginalExtended] ? originaldata[kOriginalExtended] : -1.0;
         if (rescaleddata[kExtended] != correctval) {
            fprintf(stderr, "RescaleLayerTestProbe \"%s\": RescaleLayer \"%s\", rank %d, restricted neuron %d has value %f instead of expected %f\n.",
                  this->getName(), targetRescaleLayer->getName(), parent->columnId(), k, rescaleddata[kExtended], correctval);
            status = PV_FAILURE;
         }
      }
   }
   else {
      assert(0);  // All allowable rescaleMethod values are handled above.
   }
   if (status == PV_FAILURE) {
      exit(EXIT_FAILURE);
   }
   printf("RescaleLayer \"%s\": success at time %f!\n", targetRescaleLayer->getName(), timed);
   return status;
}

bool RescaleLayerTestProbe::colinear(int nx, int ny, int ystrideA, int ystrideB, pvadata_t const * A, pvadata_t const * B, double tolerance, double * cov, double * stdA, double * stdB) {
   int numelements = nx*ny;
   if (numelements <= 1) { return false; } // Need two or more points to be meaningful
   double amean = 0.0;
   double bmean = 0.0;
   for (int y=0; y<ny; y++) {
      for (int x=0; x<nx; x++) {
         amean += (double) A[x+ystrideA*y];
         bmean += (double) B[x+ystrideB*y];
      }
   }
   amean /= numelements;
   bmean /= numelements;
   
   double astd = 0.0;
   double bstd = 0.0;
   double covariance = 0.0;
   for (int y=0; y<ny; y++) {
      for (int x=0; x<nx; x++) {
         double d1 = ((double) A[x+ystrideA*y] - amean);
         astd += d1*d1;
         double d2 = ((double) B[x+ystrideB*y] - bmean);
         bstd += d2*d2;
         covariance += d1*d2;
      }
   }
   astd /= numelements-1;
   bstd /= numelements-1;
   covariance /= numelements-1;
   astd = sqrt(astd);
   bstd = sqrt(bstd);
   if (cov) {*cov = covariance;}
   if (stdA) {*stdA = astd;}
   if (stdB) {*stdB = bstd;}
   return fabs(covariance - astd*bstd) <= tolerance;
}


} /* namespace PV */
