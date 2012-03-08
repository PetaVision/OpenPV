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

#include "../include/pv_common.h"
#include "../include/pv_types.h"

// Prototypes
static inline int updateV_HyPerLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh);
static inline int updateV_ANNLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh);
static inline int updateV_ANNDivInh(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t * GSynDivInh);
static inline int updateV_ANNSquaredLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh);
static inline int updateV_GenerativeLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t * GSynAux, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence);
static inline int updateV_PoolingANNLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t biasa, pvdata_t biasb);
static inline int updateV_PtwiseProductLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh);
static inline int updateV_TrainingLayer(int numNeurons, pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength);
static inline int updateV_GapLayer();
static inline int updateV_SigmoidLayer();

static inline int applyVMax_ANNLayer(int numNeurons, pvdata_t * V, pvdata_t VMax);
static inline int applyVThresh_ANNLayer(int numNeurons, pvdata_t * V, pvdata_t VMin, pvdata_t VThresh);
static inline int squareV_ANNSquaredLayer(int numNeurons, pvdata_t * V);

static inline int setActivity_HyPerLayer(int numNeurons, pvdata_t * V, pvdata_t * A, int nx, int ny, int nf, int nb);

// Definitions
static inline int updateV_HyPerLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh) {
   int k;
#ifndef PV_USE_OPEN_CL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPEN_CL
   {
      V[k] = GSynExc[k] - GSynInh[k];
   }
   return PV_SUCCESS;
}

static inline int updateV_ANNLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh) {
   int status;
   status = updateV_HyPerLayer(numNeurons, V, GSynExc, GSynInh);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, VMax);
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(numNeurons, V, VMin, VThresh);
   return status;
}

static inline int updateV_ANNDivInh(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t * GSynDivInh) {
   int k;
#ifndef PV_USE_OPEN_CL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPEN_CL
   {
      V[k] = (GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
   }
   return PV_SUCCESS;
}

static inline int updateV_ANNSquaredLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh) {
   int status;
   status = updateV_ANNLayer(numNeurons, V, GSynExc, GSynInh, VMax, VMin, VThresh);
   if( status == PV_SUCCESS ) status = squareV_ANNSquaredLayer(numNeurons, V);
   return status;
}

static inline int updateV_GenerativeLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t * GSynAux, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence) {
   int k;
#ifndef PV_USE_OPEN_CL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPEN_CL
   {
      pvdata_t dAnew = GSynExc[k] - GSynInh[k] + auxChannelCoeff*GSynAux[k] - sparsityTermCoeff*sparsitytermderivative[k];
      dAnew = persistence*dAold[k] + (1-persistence)*dAnew;
      V[k] += relaxation*dAnew;
      dAold[k] = dAnew;
   }
   applyVMax_ANNLayer(numNeurons, V, VMax);
   applyVThresh_ANNLayer(numNeurons, V, VMin, VThresh);
   return PV_SUCCESS;
}

static inline int updateV_PoolingANNLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t biasa, pvdata_t biasb) {
   int k;
#ifndef PV_USE_OPEN_CL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPEN_CL
   {
      V[k] = GSynExc[k]*GSynInh[k]*(biasa*GSynExc[k]+biasb*GSynInh[k]);
   }
   return PV_SUCCESS;
}

static inline int updateV_PtwiseProductLayer(int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh) {
   int k;
#ifndef PV_USE_OPEN_CL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPEN_CL
   {
      V[k] = GSynExc[k] * GSynInh[k];
   }
   return PV_SUCCESS;
}

static inline int updateV_TrainingLayer(int numNeurons, pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength) {
   assert(traininglabelindex>=0 &&
          traininglabelindex<numTrainingLabels);
   int n = trainingLabels[traininglabelindex];
   assert(n>=0 && n<numNeurons);
   V[trainingLabels[traininglabelindex]] = 0;
   traininglabelindex++;
   if( traininglabelindex == numTrainingLabels) traininglabelindex = 0;
   V[trainingLabels[traininglabelindex]] = strength;
   return PV_SUCCESS;
}

static inline int updateV_GapLayer() {
   // Contents of GapLayer::updateV() were marked obsolete at the time of refactoring.
   // The comment there read,
   // use LIFGap as source layer instead (LIFGap updates gap junctions more accurately)
   return PV_SUCCESS;
}

static inline int updateV_SigmoidLayer() {
   return PV_SUCCESS; // sourcelayer is responsible for updating V.
}

static inline int applyVMax_ANNLayer(int numNeurons, pvdata_t * V, pvdata_t VMax) {
   if( VMax < max_pvdata_t ) {
      for( int k=0; k<numNeurons; k++ ) {
         if(V[k] > VMax) V[k] = VMax;
      }
   }
   return PV_SUCCESS;
}

static inline int applyVThresh_ANNLayer(int numNeurons, pvdata_t * V, pvdata_t VMin, pvdata_t VThresh) {
   if( VThresh > -max_pvdata_t ) {
      pvdata_t * V = V;
      for( int k=0; k<numNeurons; k++ ) {
         if(V[k] < VThresh)
            V[k] = VMin;
      }
   }
   return PV_SUCCESS;
}

static inline int squareV_ANNSquaredLayer(int numNeurons, pvdata_t * V) {
   int k;
#ifndef PV_USE_OPEN_CL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPEN_CL
   {
      V[k] *= V[k];
   }
   return PV_SUCCESS;
}

static inline int setActivity_HyPerLayer(int numNeurons, pvdata_t * V, pvdata_t * A, int nx, int ny, int nf, int nb) {
   int k;
#ifndef PV_USE_OPEN_CL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPEN_CL
   {
      int kex = kIndexExtended(k,nx,ny,nf,nb);
      A[kex] = V[k];
   }
   return PV_SUCCESS;
}


#endif /* UPDATESTATEFUNCTIONS_H_ */
