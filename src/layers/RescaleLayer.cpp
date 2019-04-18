/*
 * RescaleLayer.cpp
 */

#include "RescaleLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

namespace PV {
RescaleLayer::RescaleLayer() { initialize_base(); }

RescaleLayer::RescaleLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

RescaleLayer::~RescaleLayer() { free(rescaleMethod); }

int RescaleLayer::initialize_base() {
   originalLayer = NULL;
   targetMax     = 1;
   targetMin     = -1;
   targetMean    = 0;
   targetStd     = 1;
   rescaleMethod = NULL;
   patchSize     = 1;
   return PV_SUCCESS;
}

int RescaleLayer::initialize(const char *name, HyPerCol *hc) {
   int status_init = CloneVLayer::initialize(name, hc);

   return status_init;
}

Response::Status
RescaleLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return CloneVLayer::communicateInitInfo(message);
   // CloneVLayer sets originalLayer and errors out if originalLayerName is not valid
}

// Rescale layer does not use the V buffer, so absolutely fine to clone off of an null V layer
void RescaleLayer::allocateV() {
   // Do nothing
}

int RescaleLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_rescaleMethod(ioFlag);
   if (strcmp(rescaleMethod, "maxmin") == 0) {
      ioParam_targetMax(ioFlag);
      ioParam_targetMin(ioFlag);
   }
   else if (strcmp(rescaleMethod, "meanstd") == 0) {
      ioParam_targetMean(ioFlag);
      ioParam_targetStd(ioFlag);
   }
   else if (strcmp(rescaleMethod, "pointmeanstd") == 0) {
      ioParam_targetMean(ioFlag);
      ioParam_targetStd(ioFlag);
   }
   else if (strcmp(rescaleMethod, "l2") == 0) {
      ioParam_patchSize(ioFlag);
   }
   else if (strcmp(rescaleMethod, "l2NoMean") == 0) {
      ioParam_patchSize(ioFlag);
   }
   else if (strcmp(rescaleMethod, "pointResponseNormalization") == 0) {
   }
   else if (strcmp(rescaleMethod, "zerotonegative") == 0) {
   }
   else if (strcmp(rescaleMethod, "softmax") == 0) {
   }
   else if (strcmp(rescaleMethod, "logreg") == 0) {
   }
   else {
      Fatal().printf(
            "RescaleLayer \"%s\": rescaleMethod does not exist. Current implemented methods are "
            "maxmin, meanstd, pointmeanstd, pointResponseNormalization, softmax, l2, l2NoMean, and "
            "logreg.\n",
            name);
   }
   return PV_SUCCESS;
}

void RescaleLayer::ioParam_rescaleMethod(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(ioFlag, name, "rescaleMethod", &rescaleMethod);
}

void RescaleLayer::ioParam_targetMax(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "rescaleMethod"));
   if (strcmp(rescaleMethod, "maxmin") == 0) {
      parent->parameters()->ioParamValue(ioFlag, name, "targetMax", &targetMax, targetMax);
   }
}

void RescaleLayer::ioParam_targetMin(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "rescaleMethod"));
   if (strcmp(rescaleMethod, "maxmin") == 0) {
      parent->parameters()->ioParamValue(ioFlag, name, "targetMin", &targetMin, targetMin);
   }
}

void RescaleLayer::ioParam_targetMean(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "rescaleMethod"));
   if ((strcmp(rescaleMethod, "meanstd") == 0) || (strcmp(rescaleMethod, "pointmeanstd") == 0)) {
      parent->parameters()->ioParamValue(ioFlag, name, "targetMean", &targetMean, targetMean);
   }
}

void RescaleLayer::ioParam_targetStd(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "rescaleMethod"));
   if ((strcmp(rescaleMethod, "meanstd") == 0) || (strcmp(rescaleMethod, "pointmeanstd") == 0)) {
      parent->parameters()->ioParamValue(ioFlag, name, "targetStd", &targetStd, targetStd);
   }
}

void RescaleLayer::ioParam_patchSize(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "rescaleMethod"));
   if (strcmp(rescaleMethod, "l2") == 0 || strcmp(rescaleMethod, "l2NoMean") == 0) {
      parent->parameters()->ioParamValue(ioFlag, name, "patchSize", &patchSize, patchSize);
   }
}

int RescaleLayer::setActivity() {
   float *activity = clayer->activity->data;
   memset(activity, 0, sizeof(float) * clayer->numExtendedAllBatches);
   return 0;
}

// GTK: changed to rescale activity instead of V
Response::Status RescaleLayer::updateState(double timef, double dt) {
   int numNeurons                = originalLayer->getNumNeurons();
   float *A                      = clayer->activity->data;
   const float *originalA        = originalLayer->getCLayer()->activity->data;
   const PVLayerLoc *loc         = getLayerLoc();
   const PVLayerLoc *locOriginal = originalLayer->getLayerLoc();
   int nbatch                    = loc->nbatch;
   // Make sure all sizes match
   assert(locOriginal->nx == loc->nx);
   assert(locOriginal->ny == loc->ny);
   assert(locOriginal->nf == loc->nf);

   for (int b = 0; b < nbatch; b++) {
      const float *originalABatch = originalA + b * originalLayer->getNumExtended();
      float *ABatch               = A + b * getNumExtended();

      if (strcmp(rescaleMethod, "maxmin") == 0) {
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

         MPI_Allreduce(
               MPI_IN_PLACE,
               &maxA,
               1,
               MPI_FLOAT,
               MPI_MAX,
               parent->getCommunicator()->communicator());
         MPI_Allreduce(
               MPI_IN_PLACE,
               &minA,
               1,
               MPI_FLOAT,
               MPI_MIN,
               parent->getCommunicator()->communicator());

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
                     ((originalABatch[kExtOriginal] - minA) / rangeA) * (targetMax - targetMin)
                     + targetMin;
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
      else if (strcmp(rescaleMethod, "meanstd") == 0) {
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

         MPI_Allreduce(
               MPI_IN_PLACE,
               &sum,
               1,
               MPI_FLOAT,
               MPI_SUM,
               parent->getCommunicator()->communicator());

         float mean = sum / originalLayer->getNumGlobalNeurons();

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

         MPI_Allreduce(
               MPI_IN_PLACE,
               &sumsq,
               1,
               MPI_FLOAT,
               MPI_SUM,
               parent->getCommunicator()->communicator());
         float std = sqrtf(sumsq / originalLayer->getNumGlobalNeurons());
         // The difference between the if and the else clauses is only in the computation of
         // A[kext], but this
         // way the std != 0.0 conditional is only evaluated once, not every time through the
         // for-loop.
         if (std != 0.0f) {
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
                     ((originalABatch[kextOriginal] - mean) * (targetStd / std) + targetMean);
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
      else if (strcmp(rescaleMethod, "l2") == 0) {
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

         MPI_Allreduce(
               MPI_IN_PLACE,
               &sum,
               1,
               MPI_FLOAT,
               MPI_SUM,
               parent->getCommunicator()->communicator());

         float mean = sum / originalLayer->getNumGlobalNeurons();

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

         MPI_Allreduce(
               MPI_IN_PLACE,
               &sumsq,
               1,
               MPI_FLOAT,
               MPI_SUM,
               parent->getCommunicator()->communicator());
         float std = sqrtf(sumsq / originalLayer->getNumGlobalNeurons());
         // The difference between the if and the else clauses is only in the computation of
         // A[kext], but this
         // way the std != 0.0 conditional is only evaluated once, not every time through the
         // for-loop.
         if (std != 0.0f) {
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
                      * (1.0f / (std * sqrtf((float)patchSize))));
            }
         }
         else {
            WarnLog() << "std of layer " << originalLayer->getName()
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
      else if (strcmp(rescaleMethod, "l2NoMean") == 0) {
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
         MPI_Allreduce(
               MPI_IN_PLACE,
               &sumsq,
               1,
               MPI_FLOAT,
               MPI_SUM,
               parent->getCommunicator()->communicator());
#endif // PV_USE_MPI

         float std = sqrt(sumsq / originalLayer->getNumGlobalNeurons());
         // The difference between the if and the else clauses is only in the computation of
         // A[kext], but this
         // way the std != 0.0 conditional is only evaluated once, not every time through the
         // for-loop.
         if (std != 0.0f) {
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
                     ((originalABatch[kextOriginal]) * (1.0f / (std * sqrtf((float)patchSize))));
            }
         }
         else {
            WarnLog() << "std of layer " << originalLayer->getName()
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
      else if (strcmp(rescaleMethod, "pointResponseNormalization") == 0) {
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
               float divisor = sqrtf(sumsq);
               // Difference in the if-part and else-part is only in the value assigned to A[kext],
               // but this way the std != 0
               // conditional does not have to be reevaluated every time through the for loop.
               // can't pragma omp parallel the for loops because it was already parallelized in the
               // outermost for-loop
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
      else if (strcmp(rescaleMethod, "pointmeanstd") == 0) {
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
               float std = sqrtf(sumsq / nf);
               // Difference in the if-part and else-part is only in the value assigned to A[kext],
               // but this way the std != 0
               // conditional does not have to be reevaluated every time through the for loop.
               // can't pragma omp parallel the for loops because it was already parallelized in the
               // outermost for-loop
               if (std != 0) {
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
                           ((originalABatch[kextOrig] - mean) * (targetStd / std) + targetMean);
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
      else if (strcmp(rescaleMethod, "softmax") == 0) {
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
               float maxvalue = FLT_MIN;
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
                  sumexpx += expf(originalABatch[kextOrig] - maxvalue);
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
                     ABatch[kext] = expf(originalABatch[kextOrig] - maxvalue) / sumexpx;
                  }
                  else {
                     ABatch[kext] = 0.0f;
                  }
                  assert(ABatch[kext] >= 0 && ABatch[kext] <= 1);
               }
            }
         }
      }
      else if (strcmp(rescaleMethod, "logreg") == 0) {
         int nx = loc->nx;
         int ny = loc->ny;
         int nf = loc->nf;
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
            ABatch[kext] = 1.0f / (1.0f + expf(originalABatch[kextOriginal]));
         }
      }
      else if (strcmp(rescaleMethod, "zerotonegative") == 0) {
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
   return Response::SUCCESS;
}

} // end namespace PV
