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

#ifndef PV_USE_OPENCL
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#else
#define pvdata_t float
#define max_pvdata_t FLT_MAX
#define PV_SUCCESS 0
#endif


#ifndef PV_USE_OPENCL
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL
#else  /* compiling with OpenCL */
#  define CL_KERNEL       __kernel
#  define CL_MEM_GLOBAL   __global
#  define CL_MEM_CONST    __constant
#  define CL_MEM_LOCAL    __local
#  include "conversions.hcl"
#endif

// Prototypes
static inline int updateV_HyPerLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh);
static inline int updateV_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh);
static inline int updateV_ANNDivInh(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, CL_MEM_GLOBAL pvdata_t * GSynDivInh);
static inline int updateV_ANNSquaredLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh);
static inline int updateV_GenerativeLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, CL_MEM_GLOBAL pvdata_t * GSynAux, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative, CL_MEM_GLOBAL pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence);
static inline int updateV_PoolingANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, pvdata_t biasa, pvdata_t biasb);
static inline int updateV_PtwiseProductLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh);
static inline int updateV_TrainingLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength);
static inline int updateV_GapLayer();
static inline int updateV_SigmoidLayer();

static inline int applyVMax_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, pvdata_t VMax);
static inline int applyVThresh_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, pvdata_t VMin, pvdata_t VThresh);
static inline int squareV_ANNSquaredLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V);
static inline int updateSparsityTermDeriv_GenerativeLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative);
static inline int updateSparsityTermDeriv_LogLatWTAGenLayer(int num_neurons, int num_features, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative);
static inline pvdata_t lateralCompetitionPenalty(CL_MEM_GLOBAL pvdata_t * V, int num_features);

//static inline int setActivity_HyPerLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * A, int nx, int ny, int nf, int nb);
static inline int setActivity_HyPerLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb);
static inline int setActivity_GenerativeLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb, pvdata_t activity_threshold);
static inline int setActivity_IncrementLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * Vprev, int nx, int ny, int nf, int nb);
#ifndef PV_USE_OPENCL
//opencl can't use class objects like PVLayerLoc, but I don't see an easy way to get around this, but I'll come back to it later
static inline int setActivity_GapLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb, const PVLayerLoc * src_loc, bool src_spiking, unsigned int src_num_active, unsigned int * src_active_indices);
#endif //PV_USE_OPENCL
static inline int setActivity_SigmoidLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag);

static inline int resetGSynBuffers_HyPerLayer(int num_neurons, int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int resetGSynBuffers_SigmoidLayer();
// Definitions
static inline int updateV_HyPerLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k] - GSynInh[k];
   }
   return PV_SUCCESS;
}

static inline int updateV_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh) {
   int status;
   status = updateV_HyPerLayer(numNeurons, V, GSynExc, GSynInh);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, VMax);
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(numNeurons, V, VMin, VThresh);
   return status;
}

static inline int updateV_ANNDivInh(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, CL_MEM_GLOBAL pvdata_t * GSynDivInh) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = (GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
   }
   return PV_SUCCESS;
}

static inline int updateV_ANNSquaredLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh) {
   int status;
   status = updateV_ANNLayer(numNeurons, V, GSynExc, GSynInh, VMax, VMin, VThresh);
   if( status == PV_SUCCESS ) status = squareV_ANNSquaredLayer(numNeurons, V);
   return status;
}

static inline int updateV_GenerativeLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, CL_MEM_GLOBAL pvdata_t * GSynAux, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative, CL_MEM_GLOBAL  pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
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

static inline int updateV_PoolingANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh, pvdata_t biasa, pvdata_t biasb) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k]*GSynInh[k]*(biasa*GSynExc[k]+biasb*GSynInh[k]);
   }
   return PV_SUCCESS;
}

static inline int updateV_PtwiseProductLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynExc, CL_MEM_GLOBAL pvdata_t * GSynInh) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k] * GSynInh[k];
   }
   return PV_SUCCESS;
}

static inline int updateV_TrainingLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength) {
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

static inline int applyVMax_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, pvdata_t VMax) {
   if( VMax < max_pvdata_t ) {
      int k=0;
#ifndef PV_USE_OPENCL
      for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
      {
         if(V[k] > VMax) V[k] = VMax;
      }
   }
   return PV_SUCCESS;
}

static inline int applyVThresh_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, pvdata_t VMin, pvdata_t VThresh) {
   if( VThresh > -max_pvdata_t ) {
      int k=0;
#ifndef PV_USE_OPENCL
      for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
      {
         if(V[k] < VThresh)
            V[k] = VMin;
      }
   }
   return PV_SUCCESS;
}

static inline int squareV_ANNSquaredLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] *= V[k];
   }
   return PV_SUCCESS;
}

static inline int updateSparsityTermDeriv_GenerativeLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<num_neurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      pvdata_t vk = V[k];
      sparsitytermderivative[k] = 2*vk/(1+vk*vk);
   }
   return PV_SUCCESS;
}
static inline int updateSparsityTermDeriv_LogLatWTAGenLayer(int num_neurons, int num_features, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<num_neurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int feature_start = k - (k % num_features);
      pvdata_t sum_across_features = 0.0f;
      for( int f=0; f<num_features; f++ ) sum_across_features += V[feature_start+f];
      pvdata_t lat_wta_expr = lateralCompetitionPenalty(&V[feature_start], num_features);
      // Each block of num_features neurons will have the same sum_across_features and latWTAexpr.
      // Can we eliminate redundant calculations?
      sparsitytermderivative[k] = 2*(sum_across_features-V[k])/(1+lat_wta_expr);
   }
   return PV_SUCCESS;
}

static inline pvdata_t lateralCompetitionPenalty(CL_MEM_GLOBAL pvdata_t * V, int num_features) {
   pvdata_t z=0;
   for( int p=0; p<num_features; p++ ) {
      for( int q=0; q<num_features; q++ ) {
         if( p!= q ) z += V[p]*V[q];
      }
   }
   return z;
}

static inline int setActivity_HyPerLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb) {
//static inline int setActivity_HyPerLayer(int num_neurons, pvdata_t * A, pvdata_t * V, int nx, int ny, int nf, int nb) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<num_neurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k,nx,ny,nf,nb);
      A[kex] = V[k];
   }
   return PV_SUCCESS;
}


static inline int setActivity_GenerativeLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb, pvdata_t activity_threshold) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<num_neurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      if( fabs(V[k]) > activity_threshold ) A[kex] = V[k];
   }
   return PV_SUCCESS;
}

static inline int setActivity_IncrementLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * Vprev, int nx, int ny, int nf, int nb) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<num_neurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      A[kex] = V[k]-Vprev[k];
   }
   return PV_SUCCESS;
}


//!!!TODO: add param in LIFGap for spikelet amplitude
#ifndef PV_USE_OPENCL
static inline int setActivity_GapLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb, const PVLayerLoc * src_loc, bool src_spiking, unsigned int src_num_active, unsigned int * src_active_indices) {
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, nb); // this copies the potential into the activity buffer

   // extended activity may not be current but this is alright since only local activity is used
   // !!! will break (non-deterministic) if layers are updated simultaneously--fix is to use datastore
   if(src_spiking) { // probably not needed since numActive will be zero for non-spiking
      unsigned int kActive;
   #ifndef PV_USE_OPENCL
      for (kActive = 0; kActive < src_num_active; kActive++) // If putting on a GPU, need src_num_active threads, not num_neurons
   #else
      kActive = get_global_id(0);
   #endif // PV_USE_OPENCL
      {
         int kGlobalRestricted = src_active_indices[kActive];
         int kLocalRestricted = localIndexFromGlobal(kGlobalRestricted, *src_loc);
         int kLocalExtended = kIndexExtended(kLocalRestricted, src_loc->nx, src_loc->ny, src_loc->nf, src_loc->nb);
         A[kLocalExtended] += 50; // add 50 mV spike to local membrane potential
      }
   }
   return PV_SUCCESS;
}
#endif //PV_USE_OPENCL

static inline int setActivity_SigmoidLayer(int num_neurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int nb, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag) {
   pvdata_t sig_scale = 1.0f;
   if( Vth > V0 ) {
      if( sigmoid_flag ) {
         sig_scale = -0.5f * log(1.0f/sigmoid_alpha - 1.0f) / (Vth - V0);   // scale to get response alpha at Vrest
      }
      else {
         sig_scale = 0.5/(Vth-V0); // threshold in the middle
      }
   }
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<num_neurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      if(!sigmoid_flag) {
         if (V[k] > 2*Vth-V0){    //  2x(Vth-V0) + V0
            A[kex] = 1.0f;
         }
         else if (V[k] < V0){
            A[kex] = 0.0f;
         }
         else{
            A[kex] = (V[k] - V0) * sig_scale;
         }
      }
      else{
         A[kex] = 1.0f / (1.0f + exp(2.0f * (V[k] - Vth)*sig_scale));
      }
      if (inverse_flag) A[kex] = 1.0f - A[kex];
   }
   return PV_SUCCESS;
}

static inline int resetGSynBuffers_HyPerLayer(int num_neurons, int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead) {
   for( int ch = 0; ch < num_channels; ch ++ ) {
      CL_MEM_GLOBAL pvdata_t * channelStart = &GSynHead[ch*num_neurons];
      int k;
   #ifndef PV_USE_OPENCL
      for( k=0; k<num_neurons; k++ )
   #else
         k = get_global_id(0);
   #endif // PV_USE_OPENCL
      {
         channelStart[k] = 0.0f;
      }
   }
   return PV_SUCCESS;
}

static inline int resetGSynBuffers_SigmoidLayer() {
   return PV_SUCCESS; // V is cloned from sourcelayer, so Sigmoid Layer doesn't use the GSynBuffers
}

#endif /* UPDATESTATEFUNCTIONS_H_ */
