/*
 * updateStateFunctions.h
 *
 * Static inline methods to be called by layers' updateState methods
 *
 *  Created on: Mar 7, 2012
 *      Author: pschultz
 */
#ifndef UPDATESTATEFUNCTIONS_H_
#define UPDATESTATEFUNCTIONS_H_

#ifndef PV_USE_CUDA
#include "include/pv_types.h"
#include "utils/conversions.hpp"
#endif // PV_USE_CUDA

#include "include/pv_common.h"
#include <float.h>

#ifndef PV_USE_CUDA
#define KERNEL inline
#define MEM_GLOBAL
#else
#define KERNEL __device__
#define MEM_GLOBAL
#define getIndex() (blockIdx.x * blockDim.x) + threadIdx.x
#include "cudakernels/conversions.hcu"
#define CHANNEL_EXC 0
#endif

// Prototypes
KERNEL
int applyGSyn_HyPerLCALayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      float dt_tau,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

KERNEL
int applyGSyn_HyPerLCALayer2(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      double *dtAdapt,
      float tau,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

KERNEL
int applyGSyn_ISTALayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      double *dtAdapt,
      float tau,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      float VThresh);

KERNEL
int applyGSyn_ISTALayer2(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      double *dtAdapt,
      float tau,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      float VThresh);

KERNEL
int updateV_HyPerLCALayer(
      int nbatch,
      int numNeurons,
      int numChannels,
      float *V,
      float *GSynHead,
      float *activity,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      double *dtAdapt,
      float tau,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

KERNEL
int updateV_MomentumLCALayer(
      int nbatch,
      int numNeurons,
      int numChannels,
      float *V,
      float *GSynHead,
      float *activity,
      float *prevDrive,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      double *dtAdapt,
      float tau,
      float LCAMomentumRate,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

KERNEL
int updateV_ISTALayer(
      int nbatch,
      int numNeurons,
      float *V,
      float *GSynHead,
      float *activity,
      float VThresh,
      double *dtAdapt,
      float tau,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      int numChannels);

KERNEL
int setActivity_HyPerLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *A,
      MEM_GLOBAL float *V,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

KERNEL
int setActivity_PtwiseLinearTransferLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *A,
      MEM_GLOBAL float const *V,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      int numVertices,
      float const *verticesV,
      float const *verticesA,
      float const *slopes);

// Definitions
KERNEL
int applyGSyn_HyPerLCALayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      double *dtAdapt,
      float tau,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int kbatch;
   MEM_GLOBAL float *GSynError = &GSynHead[0 * nbatch * numNeurons]; // weighted input
#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b                            = kbatch / numNeurons;
      int k                            = kbatch % numNeurons;
      float exp_tau                    = (float)exp(-dtAdapt[b] / (double)tau);
      MEM_GLOBAL float *VBatch         = V + b * numNeurons;
      MEM_GLOBAL float *GSynErrorBatch = GSynError + b * numNeurons;
      // Activity extended
      MEM_GLOBAL float *activityBatch = activity + b * (nx + rt + lt) * (ny + up + dn) * nf;
      int kex                         = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      VBatch[k]                       = exp_tau * VBatch[k]
                  + (1.0f - exp_tau) * (GSynErrorBatch[k] + selfInteract * activityBatch[kex]);
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_HyPerLCALayer2(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      double *dtAdapt,
      float tau,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int kbatch;
   MEM_GLOBAL float *GSynError  = &GSynHead[0 * nbatch * numNeurons]; // weighted input
   MEM_GLOBAL float *GSynError2 = &GSynHead[1 * nbatch * numNeurons]; // weighted input

#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;

      float exp_tau                     = (float)exp(-dtAdapt[b] / (double)tau);
      MEM_GLOBAL float *VBatch          = V + b * numNeurons;
      MEM_GLOBAL float *GSynErrorBatch  = GSynError + b * numNeurons;
      MEM_GLOBAL float *GSynError2Batch = GSynError2 + b * numNeurons;
      // Activity extended
      MEM_GLOBAL float *activityBatch = activity + b * (nx + rt + lt) * (ny + up + dn) * nf;

      int kex   = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      VBatch[k] = exp_tau * VBatch[k]
                  + (1.0f - exp_tau) * (GSynErrorBatch[k] - GSynError2Batch[k]
                                        + selfInteract * activityBatch[kex]);
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_MomentumLCALayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      MEM_GLOBAL float *prevDrive,
      double *dtAdapt,
      float tau,
      float LCAMomentumRate,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int kbatch;
   MEM_GLOBAL float *GSynError = &GSynHead[0 * nbatch * numNeurons]; // weighted input
#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b                            = kbatch / numNeurons;
      int k                            = kbatch % numNeurons;
      float exp_tau                    = expf((float)-dtAdapt[b] / tau);
      MEM_GLOBAL float *VBatch         = V + b * numNeurons;
      MEM_GLOBAL float *GSynErrorBatch = GSynError + b * numNeurons;
      MEM_GLOBAL float *prevDriveBatch = prevDrive + b * numNeurons;
      // Activity extended
      MEM_GLOBAL float *activityBatch = activity + b * (nx + rt + lt) * (ny + up + dn) * nf;
      int kex                         = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      // Calculate current drive
      float currentDrive =
            (1.0f - exp_tau) * (GSynErrorBatch[k] + selfInteract * activityBatch[kex]);
      // Accumulate into VBatch with decay and momentum
      VBatch[k] = exp_tau * VBatch[k] + currentDrive + LCAMomentumRate * prevDriveBatch[k];
      // Update momentum buffer
      prevDriveBatch[k] = currentDrive;
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_MomentumLCALayer2(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      MEM_GLOBAL float *prevDrive,
      double *dtAdapt,
      float tau,
      float LCAMomentumRate,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int kbatch;
   MEM_GLOBAL float *GSynError  = &GSynHead[0 * nbatch * numNeurons]; // weighted input
   MEM_GLOBAL float *GSynError2 = &GSynHead[1 * nbatch * numNeurons]; // weighted input

#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;

      float exp_tau                     = expf((float)-dtAdapt[b] / tau);
      MEM_GLOBAL float *VBatch          = V + b * numNeurons;
      MEM_GLOBAL float *GSynErrorBatch  = GSynError + b * numNeurons;
      MEM_GLOBAL float *GSynError2Batch = GSynError2 + b * numNeurons;
      MEM_GLOBAL float *prevDriveBatch  = prevDrive + b * numNeurons;
      // Activity extended
      MEM_GLOBAL float *activityBatch = activity + b * (nx + rt + lt) * (ny + up + dn) * nf;

      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);

      float currentDrive = (1.0f - exp_tau) * ((GSynErrorBatch[k] - GSynError2Batch[k])
                                               + selfInteract * activityBatch[kex]);
      VBatch[k]         = exp_tau * VBatch[k] + currentDrive + LCAMomentumRate * prevDriveBatch[k];
      prevDriveBatch[k] = currentDrive;
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_ISTALayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      double *dtAdapt,
      float tau,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      float VThresh) {
   int kbatch;
   MEM_GLOBAL float *GSynError = &GSynHead[CHANNEL_EXC * nbatch * numNeurons]; // weighted input
#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b                            = kbatch / numNeurons;
      int k                            = kbatch % numNeurons;
      MEM_GLOBAL float *VBatch         = V + b * numNeurons;
      MEM_GLOBAL float *GSynErrorBatch = GSynError + b * numNeurons;
      // Activity extended
      MEM_GLOBAL float *activityBatch = activity + b * (nx + rt + lt) * (ny + up + dn) * nf;
      int kex                         = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      float sign                      = 0.0f;
      if (activityBatch[kex] != 0.0f) {
         sign = activityBatch[kex] / fabsf(activityBatch[kex]);
      }
      VBatch[k] += ((float)dtAdapt[b] / tau) * (GSynErrorBatch[k] - (VThresh * sign));
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_ISTALayer2(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      double *dtAdapt,
      float tau,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      float VThresh) {
   int kbatch;
   MEM_GLOBAL float *GSynError  = &GSynHead[0 * nbatch * numNeurons]; // weighted input
   MEM_GLOBAL float *GSynError2 = &GSynHead[1 * nbatch * numNeurons]; // weighted input

#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;

      MEM_GLOBAL float *VBatch          = V + b * numNeurons;
      MEM_GLOBAL float *GSynErrorBatch  = GSynError + b * numNeurons;
      MEM_GLOBAL float *GSynError2Batch = GSynError2 + b * numNeurons;
      // Activity extended
      MEM_GLOBAL float *activityBatch = activity + b * (nx + rt + lt) * (ny + up + dn) * nf;

      int kex    = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      float sign = 0.0f;
      if (activityBatch[kex] != 0.0f) {
         sign = activityBatch[kex] / fabsf(activityBatch[kex]);
      }
      VBatch[k] += ((float)dtAdapt[b] / tau)
                   * ((GSynErrorBatch[k] - GSynError2Batch[k]) - (VThresh * sign));
   }
   return PV_SUCCESS;
}

KERNEL
int updateV_HyPerLCALayer(
      int nbatch,
      int numNeurons,
      int numChannels,
      float *V,
      float *GSynHead,
      float *activity,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      double *dtAdapt,
      float tau,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int status = PV_SUCCESS;
   if (numChannels == 2) {
      if (status == PV_SUCCESS)
         status = applyGSyn_HyPerLCALayer2(
               nbatch,
               numNeurons,
               V,
               GSynHead,
               activity,
               dtAdapt,
               tau,
               selfInteract,
               nx,
               ny,
               nf,
               lt,
               rt,
               dn,
               up);
   }
   else if (numChannels == 1) {
      if (status == PV_SUCCESS)
         status = applyGSyn_HyPerLCALayer(
               nbatch,
               numNeurons,
               V,
               GSynHead,
               activity,
               dtAdapt,
               tau,
               selfInteract,
               nx,
               ny,
               nf,
               lt,
               rt,
               dn,
               up);
   }

   if (status == PV_SUCCESS) {
      status = setActivity_PtwiseLinearTransferLayer(
            nbatch,
            numNeurons,
            activity,
            V,
            nx,
            ny,
            nf,
            lt,
            rt,
            dn,
            up,
            numVertices,
            verticesV,
            verticesA,
            slopes);
   }
   return status;
}

KERNEL
int updateV_MomentumLCALayer(
      int nbatch,
      int numNeurons,
      int numChannels,
      float *V,
      float *GSynHead,
      float *activity,
      float *prevDrive,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      double *dtAdapt,
      float tau,
      float LCAMomentumRate,
      float selfInteract,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int status = PV_SUCCESS;
   if (numChannels == 2) {
      if (status == PV_SUCCESS)
         status = applyGSyn_MomentumLCALayer2(
               nbatch,
               numNeurons,
               V,
               GSynHead,
               activity,
               prevDrive,
               dtAdapt,
               tau,
               LCAMomentumRate,
               selfInteract,
               nx,
               ny,
               nf,
               lt,
               rt,
               dn,
               up);
   }
   else if (numChannels == 1) {
      if (status == PV_SUCCESS)
         status = applyGSyn_MomentumLCALayer(
               nbatch,
               numNeurons,
               V,
               GSynHead,
               activity,
               prevDrive,
               dtAdapt,
               tau,
               LCAMomentumRate,
               selfInteract,
               nx,
               ny,
               nf,
               lt,
               rt,
               dn,
               up);
   }

   if (status == PV_SUCCESS) {
      status = setActivity_PtwiseLinearTransferLayer(
            nbatch,
            numNeurons,
            activity,
            V,
            nx,
            ny,
            nf,
            lt,
            rt,
            dn,
            up,
            numVertices,
            verticesV,
            verticesA,
            slopes);
   }
   return status;
}

KERNEL
int updateV_ISTALayer(
      int nbatch,
      int numNeurons,
      float *V,
      float *GSynHead,
      float *activity,
      float VThresh,
      double *dtAdapt,
      float tau,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      int numChannels) {
   int status = PV_SUCCESS;
   if (numChannels == 2) {
      if (status == PV_SUCCESS)
         status = applyGSyn_ISTALayer2(
               nbatch,
               numNeurons,
               V,
               GSynHead,
               activity,
               dtAdapt,
               tau,
               nx,
               ny,
               nf,
               lt,
               rt,
               dn,
               up,
               VThresh);
   }
   else if (numChannels == 1) {
      if (status == PV_SUCCESS)
         status = applyGSyn_ISTALayer(
               nbatch,
               numNeurons,
               V,
               GSynHead,
               activity,
               dtAdapt,
               tau,
               nx,
               ny,
               nf,
               lt,
               rt,
               dn,
               up,
               VThresh);
   }
   if (status == PV_SUCCESS)
      status = setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   return status;
}

KERNEL
int updateV_SigmoidLayer() {
   return PV_SUCCESS; // sourcelayer is responsible for updating V.
}

KERNEL
int setActivity_HyPerLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *A,
      MEM_GLOBAL float *V,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int kbatch;
#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b                    = kbatch / numNeurons;
      int k                    = kbatch % numNeurons;
      MEM_GLOBAL float *ABatch = A + b * ((nx + lt + rt) * (ny + up + dn) * nf);
      MEM_GLOBAL float *VBatch = V + b * numNeurons;
      int kex                  = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      ABatch[kex]              = VBatch[k];
   }
   return PV_SUCCESS;
}

KERNEL
int setActivity_PtwiseLinearTransferLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *A,
      MEM_GLOBAL float const *V,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      int numVertices,
      float const *verticesV,
      float const *verticesA,
      float const *slopes) {
   int kbatch;
   int last = numVertices - 1;
#ifndef PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_CUDA
   {
      int b               = kbatch / numNeurons;
      int k               = kbatch % numNeurons;
      float const *VBatch = V + b * numNeurons;
      float *ABatch       = A + b * (nx + lt + rt) * (ny + up + dn) * nf;
      int kex             = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      int v;
      float potential = VBatch[k];
      float activity  = 0.0f;

      if (potential < verticesV[0]) {
         activity = verticesA[0] + slopes[0] * (potential - verticesV[0]);
      }
      else if (potential >= verticesV[last]) {
         activity = verticesA[last] + slopes[numVertices] * (potential - verticesV[last]);
      }
      else {
         for (v = 0; v < last; v++) {
            if (potential < verticesV[v]) {
               break; // makes the jumps continuous from the right.  TODO: allow user control over
               // value at jump
            }
            if (potential == verticesV[v]) {
               activity = verticesA[v];
            }
            else if (potential > verticesV[v] && potential < verticesV[v + 1]) {
               activity = verticesA[v] + slopes[v + 1] * (potential - verticesV[v]);
            }
         }
      }
      ABatch[kex] = activity;
   }
   return PV_SUCCESS;
}

#endif /* UPDATESTATEFUNCTIONS_H_ */
