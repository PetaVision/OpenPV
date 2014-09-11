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
#include "../utils/conversions.h"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#else
#define pvdata_t float
#define max_pvdata_t FLT_MAX
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
#  include "../kernels/conversions.hcl"
#  define CHANNEL_EXC   0
#  define CHANNEL_INH   1
#  define CHANNEL_INHB  2
#  define CHANNEL_GAP   3
#endif

// Prototypes
static inline int applyGSyn_HyPerLayer1Channel(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int applyGSyn_HyPerLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int applyGSyn_LabelErrorLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead, int nx, int ny, int nf, int lt, int rt, int dn, int up, int isBinary);
static inline int updateV_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
      int ny, int nf, int lt, int rt, int dn, int up);
static inline int updateV_AccumulateLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
      int ny, int nf, int lt, int rt, int dn, int up);
static inline int updateV_ANNErrorLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx,
      int ny, int nf, int lt, int rt, int dn, int up, float errScale);
static inline int updateV_LabelErrorLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx,
      int ny, int nf, int lt, int rt, int dn, int up, float errScale, int isBinary);
static inline int updateV_ANNLabelLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx,
      int ny, int nf, int lt, int rt, int dn, int up);
static inline int applyGSyn_HyPerLCALayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead,
      CL_MEM_GLOBAL pvdata_t * activity, pvdata_t dt_tau, pvdata_t selfInteract, 
      int nx, int ny, int nf, int lt, int rt, int dn, int up);
static inline int applyGSyn_HyPerLCALayer2(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead,
      CL_MEM_GLOBAL pvdata_t * activity, pvdata_t dt_tau, pvdata_t selfInteract,
      int nx, int ny, int nf, int lt, int rt, int dn, int up);
static inline int applyGSyn_ANNWhitenedLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int updateV_HyPerLCALayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
      pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth,
      pvdata_t dt_tau, 
      pvdata_t selfInteract, int nx, int ny, int nf, int lt, int rt, int dn, int up, int numChannels);
static inline int updateV_ANNWhitenedLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead,
      CL_MEM_GLOBAL pvdata_t * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth,
      int nx, int ny, int nf, int lt, int rt, int dn, int up);
static inline int updateV_ANNDivInh(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int updateV_ANNSquaredLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int updateV_GenerativeLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * dV, CL_MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, pvdata_t relaxation, int nx, int ny, int nf, int lt, int rt, int dn, int up);
static inline int updateV_PoolingANNLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead,
      pvdata_t biasa, pvdata_t biasb);
static inline int updateV_PtwiseProductLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int updateV_TrainingLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, int numTrainingLabels, int * trainingLabels,
      int traininglabelindex, int strength);
static inline int updateV_GapLayer();
static inline int updateV_SigmoidLayer();
static inline int update_dV_GenerativeLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead,
      CL_MEM_GLOBAL pvdata_t * sparsitytermderivative,
      CL_MEM_GLOBAL pvdata_t * dAold, pvdata_t AMax, pvdata_t AMin,
      pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff,
      pvdata_t sparsityTermCoeff, pvdata_t persistence);
static inline int applyVMax_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      pvdata_t AMax, CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
static inline int applyVThresh_ANNLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, pvdata_t VWidth, CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
static inline int applyVThresh_ANNErrorLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
static inline int applyV_ANNLabelLayer(int numNeurons,
                                       CL_MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t AMax, pvdata_t VThresh,
                                       CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
                                       int nf, int lt, int rt, int dn, int up);
static inline int squareV_ANNSquaredLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V);
static inline int updateSparsityTermDeriv_GenerativeLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * sparsitytermderivative);
static inline int updateSparsityTermDeriv_LogLatWTAGenLayer(int numNeurons,
      int num_features, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * sparsitytermderivative);
static inline pvdata_t lateralCompetitionPenalty(CL_MEM_GLOBAL pvdata_t * V,
      int num_features);

static inline int setActivity_HyPerLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
static inline int setActivity_AccumulateLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
static inline int setActivity_GenerativeLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up, pvdata_t activity_threshold);
static inline int setActivity_IncrementLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * Vprev, int nx, int ny, int nf, int lt, int rt, int dn, int up);
static inline int setActivity_GapLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up, int orig_lt, int orig_rt, int orig_dn, int orig_up, CL_MEM_GLOBAL pvdata_t * active, float ampSpiklet);
//#ifndef PV_USE_OPENCL
//static inline int setActivity_GapLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, int orig_lt, int orig_rt, int orig_dn, int orig_up, const PVLayerLoc * src_loc, bool src_spiking, unsigned int src_num_active, unsigned int * src_active_indices);
//#endif //PV_USE_OPENCL


static inline int resetGSynBuffers_HyPerLayer(int numNeurons, int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead);
static inline int resetGSynBuffers_SigmoidLayer();

//Gaussian function prototype TODO: maybe this can go somewhere else?
//static inline float calcGausDist(float xVal, float height, float mean, float sigma);
//
//static inline float calcGausDist(float xVal, float height, float mean, float sigma){
//   return height * exp(-(pow(xVal-mean, 2)/(2*pow(sigma, 2))));
//}


// Definitions
static inline int applyGSyn_HyPerLayer1Channel(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*numNeurons];
//   CL_MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*numNeurons];
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k]; // - GSynInh[k];
   }
   return PV_SUCCESS;
}

static inline int applyGSyn_HyPerLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*numNeurons];
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k] - GSynInh[k];
   }
   return PV_SUCCESS;
}

static inline int applyGSyn_LabelErrorLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead, int nx, int ny, int nf, int lt, int rt, int dn, int up, int isBinary) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*numNeurons];
   
   if(isBinary > 0){
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
      for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
      {
         V[k] = GSynExc[k] - GSynInh[k];
         if (GSynExc[k]>0){ // target label is positive
            V[k] = V[k] > 0 ? V[k] : 0;
         }
         else {              // target label is negative
            V[k] = V[k] < 0 ? V[k] : 0;
         }
      }
   }
   else{
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
      for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
      {
         float ratio = 1;
         //Need to find maximum value of target label
         //If first feature, find ratio between target and guess feature val
         int iF = featureIndex(k, nx+lt+rt, ny+dn+up, nf);
         if(iF == 0){
            float maxTargetVal = GSynExc[k];
            int maxIdx = k;
            //Find max value in feature space
            for(int iif = 1; iif < nf; iif++){
               if(GSynExc[k+iif] > maxTargetVal){
                  maxTargetVal = GSynExc[k+iif];
                  maxIdx = k+iif;
               }
            }
            //Find ratio
            //if target label is positive and guess is over target
            if(maxTargetVal > 0 && GSynInh[maxIdx] > maxTargetVal){
               ratio = maxTargetVal / GSynInh[maxIdx];
            }
            else{
               ratio = 1;
            }
         }
         //Calculate V value based on target and rescaled guess
         V[k] = GSynExc[k] - (GSynInh[k] * ratio);
         //If target label is negative, and guess is lower than target label, err = 0
         if (GSynExc[k] < 0){
            V[k] = V[k] < 0 ? V[k] : 0;
         }
      }
   }
   return PV_SUCCESS;
}

static inline int applyGSyn_HyPerLCALayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead,
      CL_MEM_GLOBAL pvdata_t * activity, pvdata_t dt_tau, pvdata_t selfInteract, 
      int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int k;
   float exp_tau = exp(-dt_tau);
   CL_MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * numNeurons]; // weighted input
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for (k = 0; k < numNeurons; k++)
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      //V[k] = V[k] + dt_tau * (GSynError[k] - V[k] + activity[kex]);
      //      if (selfInteract){
         V[k] = exp_tau * V[k] + (1 - exp_tau) * (GSynError[k] + selfInteract * activity[kex]);}
   //else {
   //      V[k] = exp_tau * V[k] + (1 - exp_tau) * GSynError[k];}
//}
   return PV_SUCCESS;
}

static inline int calcGSyn_Mean_StdDev(int numNeurons, CL_MEM_GLOBAL pvdata_t * GSynHead,
      CL_MEM_GLOBAL double * GSyn_Mean, CL_MEM_GLOBAL double * GSyn_StdDev) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * numNeurons]; // weighted input
   *GSyn_Mean = 0;
   *GSyn_StdDev = 0;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for (k = 0; k < numNeurons; k++)
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      *GSyn_Mean += GSynError[k];
      *GSyn_StdDev += GSynError[k] * GSynError[k];
   }
   return PV_SUCCESS;
}

static inline int applyGSyn_HyPerLCALayer2(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead,
      CL_MEM_GLOBAL pvdata_t * activity, pvdata_t dt_tau, pvdata_t selfInteract, 
      int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int k;
   float exp_tau = exp(-dt_tau);
   CL_MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * numNeurons]; // weighted input
   CL_MEM_GLOBAL pvdata_t * GSynError2 = &GSynHead[1 * numNeurons]; // weighted input
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for (k = 0; k < numNeurons; k++)
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      //V[k] = V[k] + dt_tau * (GSynError[k] - GSynError2[k] - V[k] + activity[kex]);
      //if (selfInteract){
         V[k] = exp_tau * V[k] + (1 - exp_tau) * (GSynError[k] - GSynError2[k] + selfInteract * activity[kex]);}
   //else {
   //      V[k] = exp_tau * V[k] + (1 - exp_tau) * (GSynError[k]- GSynError2[k]);}
   //}
   return PV_SUCCESS;
}

static inline int applyGSyn_ANNWhitenedLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead)
{
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynInput = &GSynHead[0*numNeurons]; // un-whitened input
   CL_MEM_GLOBAL pvdata_t * GSynAveInput = &GSynHead[1*numNeurons]; // un-whitened input
   CL_MEM_GLOBAL pvdata_t * GSynAveSquaredInput = &GSynHead[2*numNeurons]; // un-whitened input
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      // set mean to zero and standard deviation to one over patch window
      V[k] = (GSynInput[k] - GSynAveInput[k]) / (sqrt(GSynAveSquaredInput[k] - GSynAveInput[k]*GSynAveInput[k]) + FLT_MIN);
   }
   return PV_SUCCESS;
}

static inline int updateV_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
        int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
        pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
        int ny, int nf, int lt, int rt, int dn, int up) {
   int status;
   if (num_channels==1) {
      status = applyGSyn_HyPerLayer1Channel(numNeurons, V, GSynHead);
   }
   else {
      status = applyGSyn_HyPerLayer(numNeurons, V, GSynHead);
   }
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
           applyVThresh_ANNLayer(numNeurons, V, AMin, VThresh, AShift, VWidth, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}

static inline int updateV_AccumulateLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
        int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
        pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
        int ny, int nf, int lt, int rt, int dn, int up) {
   int status;
   if (num_channels==1) {
      status = applyGSyn_HyPerLayer1Channel(numNeurons, V, GSynHead);
   }
   else {
      status = applyGSyn_HyPerLayer(numNeurons, V, GSynHead);
   }
   if(status == PV_SUCCESS) status = setActivity_AccumulateLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
           applyVThresh_ANNLayer(numNeurons, V, AMin, VThresh, AShift, VWidth, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}

static inline int updateV_HyPerLCALayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity,
      pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth,
      pvdata_t dt_tau, 
      pvdata_t selfInteract, int nx, int ny, int nf, int lt, int rt, int dn, int up, int numChannels)
{
   int status = PV_SUCCESS;
   if (numChannels == 2){
      if( status == PV_SUCCESS ) status =
            applyGSyn_HyPerLCALayer2(numNeurons, V, GSynHead, activity, dt_tau, selfInteract, nx, ny, nf, lt, rt, dn, up);
   }
   else if (numChannels == 1){
      if( status == PV_SUCCESS ) status =
            applyGSyn_HyPerLCALayer(numNeurons, V, GSynHead, activity, dt_tau, selfInteract, nx, ny, nf, lt, rt, dn, up);
   }
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
         applyVThresh_ANNLayer(numNeurons, V, AMin, VThresh, AShift, VWidth, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}

static inline int updateV_ANNErrorLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx, int ny, int nf, int lt, int rt, int dn, int up, float errScale)
{
   int status;
   status = applyGSyn_HyPerLayer(numNeurons, V, GSynHead);
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for(int i = 0; i < numNeurons; i++){
       V[i] *= errScale;
   }
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
         applyVThresh_ANNErrorLayer(numNeurons, V, AMin, VThresh, AShift, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}

static inline int updateV_LabelErrorLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx, int ny, int nf, int lt, int rt, int dn, int up, float errScale, int isBinary)
{
   int status;
   status = applyGSyn_LabelErrorLayer(numNeurons, V, GSynHead, nx, ny, nf, lt, rt, dn, up, isBinary);
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for(int i = 0; i < numNeurons; i++){
       V[i] *= errScale;
   }
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
         applyVThresh_ANNErrorLayer(numNeurons, V, AMin, VThresh, AShift, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}

static inline int updateV_ANNLabelLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   int status;
   status = applyGSyn_HyPerLayer(numNeurons, V, GSynHead);
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
                                  applyV_ANNLabelLayer(numNeurons, V, AMin, AMax, VThresh, activity, nx, ny, nf, lt, rt, dn, up);

   return status;
}

static inline int updateV_ANNWhitenedLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   int status;
   status = applyGSyn_ANNWhitenedLayer(numNeurons, V, GSynHead);
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
         applyVThresh_ANNLayer(numNeurons, V, AMin, VThresh, AShift, VWidth, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}

static inline int updateV_ANNDivInh(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynDivInh = &GSynHead[CHANNEL_INHB*numNeurons];
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = (GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
   }
   return PV_SUCCESS;
}

static inline int updateV_ANNSquaredLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead) {
   int status;
   status = applyGSyn_HyPerLayer1Channel(numNeurons, V, GSynHead);
//   status = updateV_ANNLayer(numNeurons, V, GSynHead, activity, AMax, AMin, VThresh,
//         0.0f, nx, ny, nf, lt, rt, dn, up);
   if (status == PV_SUCCESS)
      status = squareV_ANNSquaredLayer(numNeurons, V);
   return status;
}

static inline int updateV_GenerativeLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * dV, CL_MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, pvdata_t relaxation, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for (k = 0; k < numNeurons; k++)
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] += relaxation * dV[k];
   }
   applyVMax_ANNLayer(numNeurons, V, AMax, V, nx, ny, nf, lt, rt, dn, up);
   applyVThresh_ANNLayer(numNeurons, V, AMin, VThresh, AShift, VWidth, V, nx, ny, nf, lt, rt, dn, up);
   return PV_SUCCESS;
}

static inline int updateV_PoolingANNLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead,
      pvdata_t biasa, pvdata_t biasb) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC * numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH * numNeurons];
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for (k = 0; k < numNeurons; k++)
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k] * GSynInh[k]
                                  * (biasa * GSynExc[k] + biasb * GSynInh[k]);
   }
   return PV_SUCCESS;
}

static inline int updateV_PtwiseProductLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*numNeurons];
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k] * GSynInh[k];
   }
   return PV_SUCCESS;
}

static inline int updateV_TrainingLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, int numTrainingLabels, int * trainingLabels,
      int traininglabelindex, int strength) {
   assert(traininglabelindex>=0 && traininglabelindex<numTrainingLabels);
   int n = trainingLabels[traininglabelindex];
   assert(n>=0 && n<numNeurons);
   V[trainingLabels[traininglabelindex]] = 0;
   traininglabelindex++;
   if (traininglabelindex == numTrainingLabels)
      traininglabelindex = 0;
   if (trainingLabels[traininglabelindex] >= 0)
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

static inline int update_dV_GenerativeLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * GSynHead, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative, CL_MEM_GLOBAL  pvdata_t * dV, pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence) {
   int k;
   CL_MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*numNeurons];
   CL_MEM_GLOBAL pvdata_t * GSynAux = &GSynHead[CHANNEL_INHB*numNeurons];
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      pvdata_t dAnew = GSynExc[k] - GSynInh[k] + auxChannelCoeff*GSynAux[k] - sparsityTermCoeff*sparsitytermderivative[k];
      dV[k] = persistence*dV[k] + (1-persistence)*dAnew;
   }
   return PV_SUCCESS;
}

static inline int applyVMax_ANNLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V,
      pvdata_t AMax, CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up) {
   if (AMax < max_pvdata_t) {
      int k = 0;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
      for (k = 0; k < numNeurons; k++)
#else
         k = get_global_id(0);
#endif // PV_USE_OPENCL
      {
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (activity[kex] > AMax)
            activity[kex] = AMax;
      }
   }
   return PV_SUCCESS;
}

static inline int applyVThresh_ANNLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, pvdata_t VWidth, CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up) {
   if (VThresh > -max_pvdata_t) {
      int k = 0;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
      for (k = 0; k < numNeurons; k++)
#else
         k = get_global_id(0);
#endif // PV_USE_OPENCL
      {
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (V[k] < VThresh)
            activity[kex] = AMin;
         else if (V[k] < VThresh + VWidth)
            activity[kex] = AMin + (VThresh+VWidth-AShift-AMin)*(V[k]-VThresh)/VWidth;
         else
            activity[kex] -= AShift;
      }
   }
   return PV_SUCCESS;
}

static inline int applyVThresh_ANNErrorLayer(int numNeurons,
      CL_MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up) {
   if (VThresh > -max_pvdata_t) {
      int k = 0;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
      for (k = 0; k < numNeurons; k++)
#else
         k = get_global_id(0);
#endif // PV_USE_OPENCL
      {
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (fabs(V[k]) < VThresh)
            activity[kex] = AMin;
         else
            activity[kex] -= AShift;
      }
   }
   return PV_SUCCESS;
}

static inline int applyV_ANNLabelLayer(int numNeurons,
                                       CL_MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t AMax, pvdata_t VThresh,CL_MEM_GLOBAL pvdata_t * activity, int nx, int ny,
                                             int nf, int lt, int rt, int dn, int up) {
    if (VThresh > -max_pvdata_t) {
        int k = 0;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
        for (k = 0; k < numNeurons; k++)
#else
            k = get_global_id(0);
#endif // PV_USE_OPENCL
        {
            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
            int featureindex = featureIndex(k, nx, ny, nf) % nf;
            float factor;
            if (nf == 2)
                factor = 1.0;
            else
                factor = 255.0;
            
            if (fabs(fabs(V[k]) * factor - featureindex) < 0.00001)
                activity[kex] = AMax;
            else
                activity[kex] = AMin;
        }
    }
    return PV_SUCCESS;
}



static inline int squareV_ANNSquaredLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] *= V[k];
   }
   return PV_SUCCESS;
}

static inline int updateSparsityTermDeriv_GenerativeLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      pvdata_t vk = V[k];
      sparsitytermderivative[k] = 2*vk/(1+vk*vk);
   }
   return PV_SUCCESS;
}

static inline int updateSparsityTermDeriv_LogLatWTAGenLayer(int numNeurons, int num_features, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * sparsitytermderivative) {
#ifndef PV_USE_OPENCL
   int k;
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for (k=0; k<numNeurons/num_features; k++) {
      int feature_start = k*num_features;
      pvdata_t sum_across_features = 0.0f;
      int f;
      for (f=0; f<num_features; f++) {
         sum_across_features += V[feature_start+f];
      }
      pvdata_t lat_wta_expr = lateralCompetitionPenalty(&V[feature_start], num_features);
      for (f=0; f<num_features; f++) {
         sparsitytermderivative[k*num_features+f] = 2*(sum_across_features-V[k*num_features+f])/(1+lat_wta_expr);
      }
   }
   for (k=0; k<numNeurons; k++) {

   }
#else // PV_USE_OPENCL
   int k = get_global_id(0);
   {
      int feature_start = k - (k % num_features);
      pvdata_t sum_across_features = 0.0f;
      for( int f=0; f<num_features; f++ ) sum_across_features += V[feature_start+f];
      pvdata_t lat_wta_expr = lateralCompetitionPenalty(&V[feature_start], num_features);
      // Each block of num_features neurons will have the same sum_across_features and latWTAexpr.
      // Can we eliminate redundant calculations?
      sparsitytermderivative[k] = 2*(sum_across_features-V[k])/(1+lat_wta_expr);
   }
#endif // PV_USE_OPENCL

   return PV_SUCCESS;
}

static inline pvdata_t lateralCompetitionPenalty(CL_MEM_GLOBAL pvdata_t * V, int num_features) {
   pvdata_t z=0;
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( int p=0; p<num_features; p++ ) {
      for( int q=0; q<num_features; q++ ) {
         if( p!= q ) z += V[p]*V[q];
      }
   }
   return z;
}

static inline int setActivity_HyPerLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
//static inline int setActivity_HyPerLayer(int numNeurons, pvdata_t * A, pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k,nx,ny,nf,lt, rt, dn, up);
      A[kex] = V[k];
   }
   return PV_SUCCESS;
}

static inline int setActivity_AccumulateLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
//static inline int setActivity_HyPerLayer(int numNeurons, pvdata_t * A, pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k,nx,ny,nf,lt, rt, dn, up);
      A[kex] += V[k];
   }
   return PV_SUCCESS;
}


static inline int setActivity_GenerativeLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, pvdata_t activity_threshold) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      A[kex] = fabs(V[k])>activity_threshold ? V[k] : 0.0f;
   }
   return PV_SUCCESS;
}

static inline int setActivity_IncrementLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, CL_MEM_GLOBAL pvdata_t * Vprev, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      A[kex] = V[k]-Vprev[k];
   }
   return PV_SUCCESS;
}

static inline int setActivity_GapLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, int orig_lt, int orig_rt, int orig_dn, int orig_up, CL_MEM_GLOBAL pvdata_t * checkActive, float ampSpikelet) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k,nx,ny,nf,lt, rt, dn, up);
      int kexorig = kIndexExtended(k,nx,ny,nf,orig_lt, orig_rt, orig_dn, orig_up);
      A[kex] = V[k];
      if( checkActive[kexorig] > 0.0) A[kex] += ampSpikelet;
   }
   return PV_SUCCESS;
}

static inline int setActivity_SigmoidLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, float VthRest, float Vrest, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag, float dt) {
   pvdata_t Vth = (VthRest+Vrest)/2.0;
   pvdata_t sig_scale = -logf(1.0f/sigmoid_alpha - 1.0f)/(Vth - Vrest);
   if (!sigmoid_flag) {
      sig_scale = sig_scale/logf(3.0f);
      // If sigmoid_flag is off, A is a piecewise linear function of V, with slope of sig_scale/2 at V=Vth, truncated to have minimum value 0 and maximum value 1.
      // The log(3) term makes alpha=1/4 have the slope such that V reaches 0 at Vrest, and V reaches 1 at VthRest.  Without it, that alpha happens at 0.26894...
   }

   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      pvdata_t activity = 0.0f;
      if(!sigmoid_flag) {
         activity = 0.5f - (V[k] - Vth) * sig_scale/2;
         activity = activity < 0.0f ? 0.0f : activity;
         activity = activity > 1.0f ? 1.0f : activity;
      }
      else{
         activity = 1.0f / (1.0f + exp(2.0f * (V[k] - Vth) * sig_scale));
      }
      A[kex] = activity;
      if (inverse_flag) A[kex] = 1.0f - A[kex];
      // At this point A[kex] is in spikes per milli seconds;
      // A*dt makes activity dimensionless and timestep-independent
      // A[kex] *= dt;
      // This was moved to the strength definition of the dynamic layers

   }
   return PV_SUCCESS;
}

static inline int setActivity_MLPSigmoidLayer(int numNeurons, CL_MEM_GLOBAL pvdata_t * A, CL_MEM_GLOBAL pvdata_t * V, float linear_alpha, bool* dropout_buf, int nx, int ny, int nf, int lt, int rt, int dn, int up, float dt) {
   int k;
#ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      pvdata_t activity = 1.7159 * tanh(((float)2/3) * V[k]) + linear_alpha * V[k];
      //Set explicitly to 0 if that neuron was dropped out
      A[kex] = dropout_buf[k] ? 0 : activity;
   }
   return PV_SUCCESS;
}

static inline int resetGSynBuffers_HyPerLayer(int numNeurons, int num_channels, CL_MEM_GLOBAL pvdata_t * GSynHead) {
   for( int ch = 0; ch < num_channels; ch ++ ) {
      CL_MEM_GLOBAL pvdata_t * channelStart = &GSynHead[ch*numNeurons];
      int k;
   #ifndef PV_USE_OPENCL
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
      for( k=0; k<numNeurons; k++ )
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
