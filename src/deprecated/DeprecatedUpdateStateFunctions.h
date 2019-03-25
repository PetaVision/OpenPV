/*
 * updateStateFunctions.h
 *
 * Static inline methods to be called by layers' updateState methods
 *
 *  Created on: Mar 7, 2012
 *      Author: pschultz
 */
#ifndef DEPRECATEDUPDATESTATEFUNCTIONS_H_
#define DEPRECATEDUPDATESTATEFUNCTIONS_H_

#include "layers/updateStateFunctions.h"

// Prototypes
KERNEL
int updateV_LabelErrorLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      float errScale,
      int isBinary);

KERNEL
int updateV_ANNWhitenedLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

KERNEL
int setActivity_KmeansLayer(
      int nbatch,
      int numNeurons,
      int num_channels,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *A,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      bool trainingFlag);

// Definitions
KERNEL
int updateV_LabelErrorLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      float errScale,
      int isBinary) {
   int status;
   status = applyGSyn_LabelErrorLayer(
         nbatch, numNeurons, V, GSynHead, nx, ny, nf, lt, rt, dn, up, isBinary);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
   for (int i = 0; i < numNeurons * nbatch; i++) {
      V[i] *= errScale;
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
int updateV_ANNWhitenedLayer(
      int nbatch,
      int numNeurons,
      MEM_GLOBAL float *V,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *activity,
      int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int status;
   status = applyGSyn_ANNWhitenedLayer(nbatch, numNeurons, V, GSynHead);
   if (status == PV_SUCCESS) {
      status = setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
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

/* KmeansLayer does not use V */
KERNEL
int setActivity_KmeansLayer(
      int nbatch,
      int numNeurons,
      int num_channels,
      MEM_GLOBAL float *GSynHead,
      MEM_GLOBAL float *A,
      int nx,
      int ny,
      int nf,
      int lt,
      int rt,
      int dn,
      int up,
      bool trainingFlag) {
   MEM_GLOBAL float *GSynExc = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
   for (int b = 0; b < nbatch; b++) {
      MEM_GLOBAL float *ABatch       = A + b * ((nx + lt + rt) * (ny + up + dn) * nf);
      MEM_GLOBAL float *GSynExcBatch = GSynExc + b * numNeurons;

      for (int i = 0; i < ny; i++) {
         for (int j = 0; j < nx; j++) {
            if (trainingFlag) {
               int max = 0, maxIndex = 0;

               // look for the maximum value
               for (int k = 0; k < nf; k++) {
                  int kk = (i * nx + j) * nf + k;

                  if (GSynExcBatch[kk] > max) {
                     max      = GSynExcBatch[kk];
                     maxIndex = kk;
                  }
               }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
               for (int k = 0; k < nf; k++) {
                  int kk  = (i * nx + j) * nf + k;
                  int kex = kIndexExtended(kk, nx, ny, nf, lt, rt, dn, up);
                  if (kk == maxIndex)
                     ABatch[kex] = 1.0f;
                  else
                     ABatch[kex] = 0.0f;
               }
            }
            else {
               // compute mean
               float mean = 0.0f;

               for (int k = 0; k < nf; k++) {
                  int kk = (i * nx + j) * nf + k;
                  mean += GSynExcBatch[kk];
               }

               mean /= nf;

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
               for (int k = 0; k < nf; k++) {
                  int kk  = (i * nx + j) * nf + k;
                  int kex = kIndexExtended(kk, nx, ny, nf, lt, rt, dn, up);
                  if (GSynExcBatch[kk] >= mean) {
                     ABatch[kex] = GSynExcBatch[kk];
                  }
                  else {
                     ABatch[kex] = 0.0f;
                  }
               }
            }
         }
      }
   }

   return PV_SUCCESS;
}
#endif /* DEPRECATEDUPDATESTATEFUNCTIONS_H_ */
