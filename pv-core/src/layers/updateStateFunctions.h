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

#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
#include "../utils/conversions.h"
#include "../include/pv_types.h"
#endif // PV_USE_OPENCL

#include <float.h>
#include "../include/pv_common.h"
#include "../include/pv_datatypes.h"


#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
#  define KERNEL         static inline
#  define MEM_GLOBAL
#  define MEM_CONST
#  define MEM_LOCAL
#elif defined(PV_USE_OPENCL) /* compiling with OpenCL */
#  define KERNEL         static inline 
#  define MEM_GLOBAL   __global
#  define MEM_CONST    __constant
#  define MEM_LOCAL    __local
#  define getIndex() get_global_id(0)
#  include "../kernels/conversions.hcl"
#  define CHANNEL_EXC   0
#  define CHANNEL_INH   1
#  define CHANNEL_INHB  2
#  define CHANNEL_GAP   3
#elif defined(PV_USE_CUDA) /* compiling with cuda */
#  define KERNEL       __device__
#  define MEM_GLOBAL   
#  define MEM_CONST    
#  define MEM_LOCAL    
#  define getIndex() (blockIdx.x * blockDim.x) + threadIdx.x
#  include "../cudakernels/conversions.hcu"
#  define CHANNEL_EXC   0
#  define CHANNEL_INH   1
#  define CHANNEL_INHB  2
#  define CHANNEL_GAP   3
#endif

// Prototypes
KERNEL
int applyGSyn_HyPerLayer1Channel(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead);
KERNEL
int applyGSyn_HyPerLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead);
KERNEL
int applyGSyn_LabelErrorLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead, int nx, int ny, int nf, int lt, int rt, int dn, int up, int isBinary);
KERNEL
int updateV_PtwiseLinearTransferLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * activity,
      int numVertices, float * verticesV, float * verticesA, float * slopes,
      int nx, int ny, int nf, int lt, int rt, int dn, int up);

#ifdef OBSOLETE // Marked obsolete July 27, 2015.  Use updateV_PtwiseLinearTransferLayer() instead.
KERNEL
int updateV_ANNLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
      int ny, int nf, int lt, int rt, int dn, int up);
#endif // OBSOLETE // Marked obsolete July 27, 2015.  Use updateV_PtwiseLinearTransferLayer() instead.
#ifdef OBSOLETE // Marked obsolete July 27, 2015.  AccumulateLayer has been moved to obsolete folder.
KERNEL
int updateV_AccumulateLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
      int ny, int nf, int lt, int rt, int dn, int up);
#endif // OBSOLETE // Marked obsolete July 27, 2015.  AccumulateLayer has been moved to obsolete folder.
KERNEL
int updateV_ANNErrorLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * activity,
      int numVertices, float * verticesV, float * verticesA, float * slopes,
      int nx, int ny, int nf, int lt, int rt, int dn, int up, float errScale);
KERNEL
int updateV_LabelErrorLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * activity,
      int numVertices, float * verticesV, float * verticesA, float * slopes,
      int nx, int ny, int nf, int lt, int rt, int dn, int up, float errScale, int isBinary);
#ifdef OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.
KERNEL
int updateV_ANNLabelLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
      pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx,
      int ny, int nf, int lt, int rt, int dn, int up);
#endif // OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.
KERNEL
int applyGSyn_HyPerLCALayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
      MEM_GLOBAL pvdata_t * activity, pvdata_t dt_tau, pvdata_t selfInteract, 
      int nx, int ny, int nf, int lt, int rt, int dn, int up);
KERNEL
int applyGSyn_HyPerLCALayer2(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
      MEM_GLOBAL pvdata_t * activity, double* dtAdapt, pvdata_t tau, pvdata_t selfInteract,
      int nx, int ny, int nf, int lt, int rt, int dn, int up);
KERNEL
int applyGSyn_ISTALayer(int nbatch, int numNeurons,
			    MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
			    MEM_GLOBAL pvdata_t * activity, double* dtAdapt, pvdata_t tau,
			    int nx, int ny, int nf, int lt, int rt, int dn, int up, pvdata_t VThresh);
KERNEL
int applyGSyn_ISTALayer2(int nbatch, int numNeurons,
			     MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
			     MEM_GLOBAL pvdata_t * activity, double* dtAdapt, pvdata_t tau,
			     int nx, int ny, int nf, int lt, int rt, int dn, int up, pvdata_t VThresh);
KERNEL
int applyGSyn_ANNWhitenedLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead);

KERNEL
int updateV_HyPerLCALayer(int nbatch, int numNeurons, int numChannels, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
      int numVertices, pvpotentialdata_t * verticesV, pvadata_t * verticesA, float * slopes,
      double * dtAdapt, float tau, pvdata_t selfInteract,
      int nx, int ny, int nf, int lt, int rt, int dn, int up);

KERNEL
int updateV_ISTALayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
      pvdata_t VThresh,MEM_GLOBAL double* dtAdapt, pvdata_t tau,
      int nx, int ny, int nf, int lt, int rt, int dn, int up, int numChannels);
KERNEL
int updateV_ANNWhitenedLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvpotentialdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead,
      MEM_GLOBAL pvadata_t * activity,
      int numVertices, pvpotentialdata_t * verticesV, pvadata_t * verticesA, float * slopes,
      int nx, int ny, int nf, int lt, int rt, int dn, int up);
KERNEL
int updateV_ANNDivInh(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead);
KERNEL
int updateV_ANNSquaredLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead);
#ifdef OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int updateV_GenerativeLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * dV, MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, pvdata_t relaxation, int nx, int ny, int nf, int lt, int rt, int dn, int up);
#endif // OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int updateV_PoolingANNLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
      pvdata_t biasa, pvdata_t biasb);
KERNEL
int updateV_PtwiseProductLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead);
//KERNEL
//int updateV_TrainingLayer(int nbatch, int numNeurons,
//      MEM_GLOBAL pvdata_t * V, int numTrainingLabels, int * trainingLabels,
//      int traininglabelindex, int strength);
KERNEL
int updateV_GapLayer();
KERNEL
int updateV_SigmoidLayer();
#ifdef OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int update_dV_GenerativeLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
      MEM_GLOBAL pvdata_t * sparsitytermderivative,
      MEM_GLOBAL pvdata_t * dAold, pvdata_t AMax, pvdata_t AMin,
      pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff,
      pvdata_t sparsityTermCoeff, pvdata_t persistence);
#endif // OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
#ifdef OBSOLETE // Marked obsolete July 28, 2015.  Use setActivity_PtwiseLinearTransferLayer instead of applyVThresh_ANNLayer and applyVMax_ANNLayer
KERNEL
int applyVMax_ANNLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      pvdata_t AMax, MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
KERNEL
int applyVThresh_ANNLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, pvdata_t VWidth, MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
#endif // OBSOLETE // Marked obsolete July 28, 2015.  Use setActivity_PtwiseLinearTransferLayer instead of applyVThresh_ANNLayer and applyVMax_ANNLayer
KERNEL
int applyVThresh_ANNErrorLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
#ifdef OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.
KERNEL
int applyV_ANNLabelLayer(int nbatch, int numNeurons,
                                       MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t AMax, pvdata_t VThresh,
                                       MEM_GLOBAL pvdata_t * activity, int nx, int ny,
                                       int nf, int lt, int rt, int dn, int up);
#endif // OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.
KERNEL
int squareV_ANNSquaredLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V);
#ifdef OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int updateSparsityTermDeriv_GenerativeLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * sparsitytermderivative);
#endif // OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int updateSparsityTermDeriv_LogLatWTAGenLayer(int nbatch, int numNeurons,
      int num_features, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * sparsitytermderivative);
KERNEL
pvdata_t lateralCompetitionPenalty(MEM_GLOBAL pvdata_t * V,
      int num_features);

KERNEL
int setActivity_HyPerLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);

KERNEL
int setActivity_PtwiseLinearTransferLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up, int numVertices,
      pvdata_t * verticesV, pvdata_t * verticesA, float * slopes);

KERNEL
int setActivity_AccumulateLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up);
#ifdef OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int setActivity_GenerativeLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up, pvdata_t activity_threshold);
#endif // OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int setActivity_IncrementLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * Vprev, int nx, int ny, int nf, int lt, int rt, int dn, int up);
KERNEL
int setActivity_GapLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny,
      int nf, int lt, int rt, int dn, int up, int orig_lt, int orig_rt, int orig_dn, int orig_up, MEM_GLOBAL pvdata_t * active, float ampSpiklet);
//#ifndef PV_USE_OPENCL
//static inline int setActivity_GapLayer(int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, int orig_lt, int orig_rt, int orig_dn, int orig_up, const PVLayerLoc * src_loc, bool src_spiking, unsigned int src_num_active, unsigned int * src_active_indices);
//#endif //PV_USE_OPENCL


KERNEL
int resetGSynBuffers_HyPerLayer(int nbatch, int numNeurons, int num_channels, MEM_GLOBAL pvdata_t * GSynHead);
KERNEL
int resetGSynBuffers_PoolingIndexLayer(int nbatch, int numNeurons, int num_channels, MEM_GLOBAL float * GSynHead);
KERNEL
int resetGSynBuffers_SigmoidLayer();

//Gaussian function prototype TODO: maybe this can go somewhere else?
//static inline float calcGausDist(float xVal, float height, float mean, float sigma);
//
//static inline float calcGausDist(float xVal, float height, float mean, float sigma){
//   return height * exp(-(pow(xVal-mean, 2)/(2*pow(sigma, 2))));
//}

KERNEL
int setActivity_KmeansLayer(int nbatch, int numNeurons, int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * A,int nx,int ny, int nf, int lt, int rt, int dn, int up, bool trainingFlag);


// Definitions
KERNEL
int applyGSyn_HyPerLayer1Channel(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead) {
   int kbatch;
   //gSyn spins from slowest to fastest: [channel, batch, ny, nx, nf]
   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
#else
   kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
      MEM_GLOBAL pvdata_t* GSynExcBatch = GSynExc + b*numNeurons;
      VBatch[k] = GSynExcBatch[k]; // - GSynInh[k];
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_HyPerLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead) {
   int kbatch;
   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( kbatch=0; kbatch<numNeurons * nbatch; kbatch++ )
#else
   kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
      MEM_GLOBAL pvdata_t* GSynExcBatch = GSynExc + b*numNeurons;
      MEM_GLOBAL pvdata_t* GSynInhBatch = GSynInh + b*numNeurons;

      VBatch[k] = GSynExcBatch[k] - GSynInhBatch[k];
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_LabelErrorLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead, int nx, int ny, int nf, int lt, int rt, int dn, int up, int isBinary) {
   int kbatch;
   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
   
   if(isBinary > 0){
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
      for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
#else
      kbatch = getIndex();
#endif // PV_USE_OPENCL
      {
         int b = kbatch / numNeurons;
         int k = kbatch % numNeurons;
         MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
         MEM_GLOBAL pvdata_t* GSynExcBatch = GSynExc + b*numNeurons;
         MEM_GLOBAL pvdata_t* GSynInhBatch = GSynInh + b*numNeurons;
         VBatch[k] = GSynExcBatch[k] - GSynInhBatch[k];
         if (GSynExcBatch[k]>0){ // target label is positive
            VBatch[k] = VBatch[k] > 0 ? VBatch[k] : 0;
         }
         else {              // target label is negative
            VBatch[k] = VBatch[k] < 0 ? VBatch[k] : 0;
         }
      }
   }
   else{
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
      for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
#else
      kbatch = getIndex();
#endif // PV_USE_OPENCL
      {
         int b = kbatch / numNeurons;
         int k = kbatch % numNeurons;
         pvdata_t* VBatch = V + b*numNeurons;
         pvdata_t* GSynExcBatch = GSynExc + b*numNeurons;
         pvdata_t* GSynInhBatch = GSynInh + b*numNeurons;

         float ratio = 1;
         //Need to find maximum value of target label
         //If first feature, find ratio between target and guess feature val
         int iF = featureIndex(k, nx+lt+rt, ny+dn+up, nf);
         if(iF == 0){
            float maxTargetVal = GSynExcBatch[k];
            int maxIdx = k;
            //Find max value in feature space
            for(int iif = 1; iif < nf; iif++){
               if(GSynExcBatch[k+iif] > maxTargetVal){
                  maxTargetVal = GSynExcBatch[k+iif];
                  maxIdx = k+iif;
               }
            }
            //Find ratio
            //if target label is positive and guess is over target
            if(maxTargetVal > 0 && GSynInhBatch[maxIdx] > maxTargetVal){
               ratio = maxTargetVal / GSynInhBatch[maxIdx];
            }
            else{
               ratio = 1;
            }
         }
         //Calculate V value based on target and rescaled guess
         VBatch[k] = GSynExcBatch[k] - (GSynInhBatch[k] * ratio);
         //If target label is negative, and guess is lower than target label, err = 0
         if (GSynExcBatch[k] < 0){
            VBatch[k] = VBatch[k] < 0 ? VBatch[k] : 0;
         }
      }
   }
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_HyPerLCALayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
      MEM_GLOBAL pvdata_t * activity, double* dtAdapt, pvdata_t tau, pvdata_t selfInteract, 
      int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int kbatch;
   MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * nbatch * numNeurons]; // weighted input
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for (kbatch = 0; kbatch < numNeurons*nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      float exp_tau = exp(-dtAdapt[b]/tau);
      MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
      MEM_GLOBAL pvdata_t* GSynErrorBatch = GSynError + b*numNeurons;
      //Activity extended
      MEM_GLOBAL pvdata_t* activityBatch = activity + b*(nx+rt+lt)*(ny+up+dn)*nf;
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      //V[k] = V[k] + dt_tau * (GSynError[k] - V[k] + activity[kex]);
      //      if (selfInteract){
         VBatch[k] = exp_tau * VBatch[k] + (1 - exp_tau) * (GSynErrorBatch[k] + selfInteract * activityBatch[kex]);
   }
   //else {
   //      V[k] = exp_tau * V[k] + (1 - exp_tau) * GSynError[k];}
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_HyPerLCALayer2(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
      MEM_GLOBAL pvdata_t * activity, double* dtAdapt, pvdata_t tau, pvdata_t selfInteract, 
      int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int kbatch;
   MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * nbatch * numNeurons]; // weighted input
   MEM_GLOBAL pvdata_t * GSynError2 = &GSynHead[1 * nbatch * numNeurons]; // weighted input

#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for (kbatch = 0; kbatch < numNeurons*nbatch; kbatch++)
#else
   kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;

      float exp_tau = exp(-dtAdapt[b]/tau);
      MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
      MEM_GLOBAL pvdata_t* GSynErrorBatch = GSynError + b*numNeurons;
      MEM_GLOBAL pvdata_t* GSynError2Batch = GSynError2 + b*numNeurons;
      //Activity extended
      MEM_GLOBAL pvdata_t* activityBatch = activity + b*(nx+rt+lt)*(ny+up+dn)*nf;

      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      //V[k] = V[k] + dt_tau * (GSynError[k] - GSynError2[k] - V[k] + activity[kex]);
      //if (selfInteract){
         VBatch[k] = exp_tau * VBatch[k] + (1 - exp_tau) * (GSynErrorBatch[k] - GSynError2Batch[k] + selfInteract * activityBatch[kex]);
   }
   //else {
   //      V[k] = exp_tau * V[k] + (1 - exp_tau) * (GSynError[k]- GSynError2[k]);}
   //}
   return PV_SUCCESS;
}

KERNEL
int applyGSyn_ISTALayer(int nbatch, int numNeurons,
			    MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
			    MEM_GLOBAL pvdata_t * activity, double* dtAdapt, pvdata_t tau,
			int nx, int ny, int nf, int lt, int rt, int dn, int up, pvdata_t VThresh) {
  int kbatch;
  MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * nbatch * numNeurons]; // weighted input                                              
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
   #endif
  for (kbatch = 0; kbatch < numNeurons*nbatch; kbatch++)
#else
    kbatch = getIndex();
#endif // PV_USE_OPENCL                                                                                                                 
  {
    int b = kbatch / numNeurons;
    int k = kbatch % numNeurons;
    float exp_tau = exp(-dtAdapt[b]/tau);
    MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
    MEM_GLOBAL pvdata_t* GSynErrorBatch = GSynError + b*numNeurons;
    //Activity extended                                                                                                               
    MEM_GLOBAL pvdata_t* activityBatch = activity + b*(nx+rt+lt)*(ny+up+dn)*nf;
    int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
    //V[k] = V[k] + dt_tau * (GSynError[k] - V[k] + activity[kex]);                                                                   
    //      if (selfInteract){
    float sign;
    if (activityBatch[kex] == 0)
      sign = 0;
    else
      sign = activityBatch[kex]/fabsf(activityBatch[kex]);
    VBatch[k] += (dtAdapt[b]/tau) * (GSynErrorBatch[k] - (VThresh * sign));
  }
  //else {                                                                                                                             
  //      V[k] = exp_tau * V[k] + (1 - exp_tau) * GSynError[k];}                                                                       
  return PV_SUCCESS;
}

KERNEL
int applyGSyn_ISTALayer2(int nbatch, int numNeurons,
			     MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
			     MEM_GLOBAL pvdata_t * activity, double* dtAdapt, pvdata_t tau,
			 int nx, int ny, int nf, int lt, int rt, int dn, int up, pvdata_t VThresh) {
  int kbatch;
  MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * nbatch * numNeurons]; // weighted input                                              
  MEM_GLOBAL pvdata_t * GSynError2 = &GSynHead[1 * nbatch * numNeurons]; // weighted input                                             

#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
   #endif
  for (kbatch = 0; kbatch < numNeurons*nbatch; kbatch++)
#else
    kbatch = getIndex();
#endif // PV_USE_OPENCL                                                                                                                 
  {
    int b = kbatch / numNeurons;
    int k = kbatch % numNeurons;

    float exp_tau = exp(-dtAdapt[b]/tau);
    MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
    MEM_GLOBAL pvdata_t* GSynErrorBatch = GSynError + b*numNeurons;
    MEM_GLOBAL pvdata_t* GSynError2Batch = GSynError2 + b*numNeurons;
    //Activity extended                                                                                                               
    MEM_GLOBAL pvdata_t* activityBatch = activity + b*(nx+rt+lt)*(ny+up+dn)*nf;

    int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
    //V[k] = V[k] + dt_tau * (GSynError[k] - GSynError2[k] - V[k] + activity[kex]);                                                   
    //if (selfInteract){                                                                                                              
    //VBatch[k] = exp_tau * VBatch[k] + (1 - exp_tau) * (GSynErrorBatch[k] - GSynError2Batch[k] - (VThresh * (activityBatch[kex] > 0)));
    float sign;
    if (activityBatch[kex] == 0)
      sign = 0;
    else
      sign = activityBatch[kex]/fabsf(activityBatch[kex]);
    VBatch[k] += (dtAdapt[b]/tau) * ( (GSynErrorBatch[k] - GSynError2Batch[k]) - (VThresh * sign) );
  }
  //else {                                                                                                                             
  //      V[k] = exp_tau * V[k] + (1 - exp_tau) * (GSynError[k]- GSynError2[k]);}                                                      
  //}                                                                                                                                  
  return PV_SUCCESS;
}

KERNEL
int applyGSyn_ANNWhitenedLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead)
{
   int k;
   MEM_GLOBAL pvdata_t * GSynInput = &GSynHead[0*nbatch*numNeurons]; // un-whitened input
   MEM_GLOBAL pvdata_t * GSynAveInput = &GSynHead[1*nbatch*numNeurons]; // un-whitened input
   MEM_GLOBAL pvdata_t * GSynAveSquaredInput = &GSynHead[2*nbatch*numNeurons]; // un-whitened input
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( k=0; k<numNeurons * nbatch; k++ )
#else
   k = getIndex();
#endif // PV_USE_OPENCL
   {
      // set mean to zero and standard deviation to one over patch window
      V[k] = (GSynInput[k] - GSynAveInput[k]) / (sqrt(GSynAveSquaredInput[k] - GSynAveInput[k]*GSynAveInput[k]) + FLT_MIN);
   }
   return PV_SUCCESS;
}

KERNEL
int updateV_PtwiseLinearTransferLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
        int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * activity,
        int numVertices, float * verticesV, float * verticesA, float * slopes,
        int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   int status = PV_SUCCESS;
   if (num_channels==1) {
      status = applyGSyn_HyPerLayer1Channel(nbatch, numNeurons, V, GSynHead);
   }
   else {
      status = applyGSyn_HyPerLayer(nbatch, numNeurons, V, GSynHead);
   }
   if (status==PV_SUCCESS) {
      status = setActivity_PtwiseLinearTransferLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up, numVertices, verticesV, verticesA, slopes);
   }
   return status;
}

#ifdef OBSOLETE // Marked obsolete July 27, 2015.  Use updateV_PtwiseLinearTransferLayer() instead.
KERNEL
int updateV_ANNLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
        int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
        pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
        int ny, int nf, int lt, int rt, int dn, int up) {
   int status;
   if (num_channels==1) {
      status = applyGSyn_HyPerLayer1Channel(nbatch, numNeurons, V, GSynHead);
   }
   else {
      status = applyGSyn_HyPerLayer(nbatch, numNeurons, V, GSynHead);
   }
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
           applyVThresh_ANNLayer(nbatch, numNeurons, V, AMin, VThresh, AShift, VWidth, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(nbatch, numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}
#endif // OBSOLETE // Marked obsolete July 27, 2015.  Use updateV_PtwiseLinearTransferLayer() instead.

#ifdef OBSOLETE // Marked obsolete July 27, 2015.  AccumulateLayer has been moved to obsolete folder.
KERNEL
int updateV_AccumulateLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
        int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
        pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, int nx,
        int ny, int nf, int lt, int rt, int dn, int up) {
   int status;
   if (num_channels==1) {
      status = applyGSyn_HyPerLayer1Channel(nbatch, numNeurons, V, GSynHead);
   }
   else {
      status = applyGSyn_HyPerLayer(nbatch, numNeurons, V, GSynHead);
   }
   if(status == PV_SUCCESS) status = setActivity_AccumulateLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
           applyVThresh_ANNLayer(nbatch, numNeurons, V, AMin, VThresh, AShift, VWidth, activity, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(nbatch, numNeurons, V, AMax, activity, nx, ny, nf, lt, rt, dn, up);
   return status;
}
#endif // OBSOLETE // Marked obsolete July 27, 2015.  AccumulateLayer has been moved to obsolete folder.

KERNEL
int updateV_HyPerLCALayer(int nbatch, int numNeurons, int numChannels, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
      int numVertices, pvpotentialdata_t * verticesV, pvadata_t * verticesA, float * slopes,
      double * dtAdapt, float tau, pvdata_t selfInteract,
      int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   int status = PV_SUCCESS;
   if (numChannels == 2){
      if( status == PV_SUCCESS ) status =
            applyGSyn_HyPerLCALayer2(nbatch, numNeurons, V, GSynHead, activity, dtAdapt, tau, selfInteract, nx, ny, nf, lt, rt, dn, up);
   }
   else if (numChannels == 1){
      if( status == PV_SUCCESS ) status =
            applyGSyn_HyPerLCALayer(nbatch, numNeurons, V, GSynHead, activity, dtAdapt, tau, selfInteract, nx, ny, nf, lt, rt, dn, up);
   }

   if (status==PV_SUCCESS) {
      status = setActivity_PtwiseLinearTransferLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up, numVertices, verticesV, verticesA, slopes);
   }
   return status;
}

KERNEL
int updateV_ISTALayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
			  MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity,
			  pvdata_t VThresh, 
			  MEM_GLOBAL double* dtAdapt, pvdata_t tau,
			  int nx, int ny, int nf, int lt, int rt, int dn, int up, int numChannels)
{
  int status = PV_SUCCESS;
  if (numChannels == 2){
    if( status == PV_SUCCESS ) status =
				 applyGSyn_ISTALayer2(nbatch, numNeurons, V, GSynHead, activity, dtAdapt, tau, nx, ny, nf, lt, rt, dn, up, VThresh);
  }
  else if (numChannels == 1){
    if( status == PV_SUCCESS ) status =
				 applyGSyn_ISTALayer(nbatch, numNeurons, V, GSynHead, activity, dtAdapt, tau, nx, ny, nf, lt, rt, dn, up, VThresh);
  }
  if(status == PV_SUCCESS) status = setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
  return status;
}

KERNEL
int updateV_ANNErrorLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * activity,
      int numVertices, float * verticesV, float * verticesA, float * slopes,
      int nx, int ny, int nf, int lt, int rt, int dn, int up, float errScale)
{
   int status;
   status = applyGSyn_HyPerLayer(nbatch, numNeurons, V, GSynHead);
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for(int i = 0; i < numNeurons*nbatch; i++){
       V[i] *= errScale;
   }
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) {
      status = setActivity_PtwiseLinearTransferLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up, numVertices, verticesV, verticesA, slopes);
   }
   return status;
}

KERNEL
int updateV_LabelErrorLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * activity,
      int numVertices, float * verticesV, float * verticesA, float * slopes,
      int nx, int ny, int nf, int lt, int rt, int dn, int up, float errScale, int isBinary)
{
   int status;
   status = applyGSyn_LabelErrorLayer(nbatch, numNeurons, V, GSynHead, nx, ny, nf, lt, rt, dn, up, isBinary);
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for(int i = 0; i < numNeurons*nbatch; i++){
       V[i] *= errScale;
   }
   if (status==PV_SUCCESS) {
      status = setActivity_PtwiseLinearTransferLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up, numVertices, verticesV, verticesA, slopes);
   }
   return status;
}

#ifdef OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.
KERNEL
int updateV_ANNLabelLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   int status;
   status = applyGSyn_HyPerLayer(numNeurons, V, GSynHead);
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if( status == PV_SUCCESS ) status =
                                  applyV_ANNLabelLayer(numNeurons, V, AMin, AMax, VThresh, activity, nx, ny, nf, lt, rt, dn, up);

   return status;
}
#endif // OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.

KERNEL
int updateV_ANNWhitenedLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvpotentialdata_t * V,
      MEM_GLOBAL pvdata_t * GSynHead,
      MEM_GLOBAL pvadata_t * activity,
      int numVertices, pvpotentialdata_t * verticesV, pvadata_t * verticesA, float * slopes,
      int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   int status;
   status = applyGSyn_ANNWhitenedLayer(nbatch, numNeurons, V, GSynHead);
   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
   if (status==PV_SUCCESS) {
      status = setActivity_PtwiseLinearTransferLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up, numVertices, verticesV, verticesA, slopes);
   }
   return status;
}

KERNEL
int updateV_ANNDivInh(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead) {
   int k;
   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
   MEM_GLOBAL pvdata_t * GSynDivInh = &GSynHead[CHANNEL_INHB*nbatch*numNeurons];

#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( k=0; k<numNeurons*nbatch; k++ )
#else
   k = getIndex();
#endif // PV_USE_OPENCL
   {
      V[k] = (GSynExc[k] - GSynInh[k])/(GSynDivInh[k]+0.04);
   }
   return PV_SUCCESS;
}

KERNEL
int updateV_ANNSquaredLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead) {
   int status;
   status = applyGSyn_HyPerLayer1Channel(nbatch, numNeurons, V, GSynHead);
   if (status == PV_SUCCESS)
      status = squareV_ANNSquaredLayer(nbatch, numNeurons, V);
   return status;
}

#ifdef OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int updateV_GenerativeLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * dV, MEM_GLOBAL float * activity, pvdata_t AMax,
      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, pvdata_t relaxation, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int k;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for (k = 0; k < numNeurons*nbatch; k++)
#else
   k = getIndex();
#endif // PV_USE_OPENCL
   {
      V[k] += relaxation * dV[k];
   }
   applyVMax_ANNLayer(nbatch, numNeurons, V, AMax, V, nx, ny, nf, lt, rt, dn, up);
   applyVThresh_ANNLayer(nbatch, numNeurons, V, AMin, VThresh, AShift, VWidth, V, nx, ny, nf, lt, rt, dn, up);
   return PV_SUCCESS;
}
#endif // OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative

KERNEL
int updateV_PoolingANNLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead,
      pvdata_t biasa, pvdata_t biasb) {
   int k;
   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC * nbatch * numNeurons];
   MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH * nbatch * numNeurons];
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for (k = 0; k < numNeurons * nbatch; k++)
#else
      k = getIndex();
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k] * GSynInh[k]
                                  * (biasa * GSynExc[k] + biasb * GSynInh[k]);
   }
   return PV_SUCCESS;
}

KERNEL
int updateV_PtwiseProductLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead) {
   int k;
   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( k=0; k<numNeurons*nbatch; k++ )
#else
      k = getIndex();
#endif // PV_USE_OPENCL
   {
      V[k] = GSynExc[k] * GSynInh[k];
   }
   return PV_SUCCESS;
}

KERNEL
int updateV_GapLayer() {
   // Contents of GapLayer::updateV() were marked obsolete at the time of refactoring.
   // The comment there read,
   // use LIFGap as source layer instead (LIFGap updates gap junctions more accurately)
   return PV_SUCCESS;
}

KERNEL
int updateV_SigmoidLayer() {
   return PV_SUCCESS; // sourcelayer is responsible for updating V.
}

#ifdef OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int update_dV_GenerativeLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * sparsitytermderivative, MEM_GLOBAL  pvdata_t * dV, pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence) {
   int k;
   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*numNeurons];
   MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*numNeurons];
   MEM_GLOBAL pvdata_t * GSynAux = &GSynHead[CHANNEL_INHB*numNeurons];
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = getIndex();
#endif // PV_USE_OPENCL
   {
      pvdata_t dAnew = GSynExc[k] - GSynInh[k] + auxChannelCoeff*GSynAux[k] - sparsityTermCoeff*sparsitytermderivative[k];
      dV[k] = persistence*dV[k] + (1-persistence)*dAnew;
   }
   return PV_SUCCESS;
}
#endif // OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative

#ifdef OBSOLETE // Marked obsolete July 28, 2015.  Use setActivity_PtwiseLinearTransferLayer instead of applyVThresh_ANNLayer and applyVMax_ANNLayer
KERNEL
int applyVMax_ANNLayer(int numNeurons, MEM_GLOBAL pvdata_t * V,
      pvdata_t AMax, MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up) {
   if (AMax < max_pvadata_t) {
      int kbatch = 0;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
      #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for schedule(static)
      #endif
      for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
      kbatch = getIndex();
#endif // PV_USE_OPENCL
      {
         int b = kbatch / numNeurons;
         int k = kbatch % numNeurons;
         MEM_GLOBAL pvdata_t * activityBatch = activity + b*(nx+lt+rt)*(ny+up+dn)*nf;
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (activityBatch[kex] > AMax)
            activityBatch[kex] = AMax;
      }
   }
   return PV_SUCCESS;
}

KERNEL
int applyVThresh_ANNLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, pvdata_t VWidth, MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up) {
   if (VThresh > -max_pvvdata_t) {
      int kbatch = 0;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
      for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
      kbatch = getIndex();;
#endif // PV_USE_OPENCL
      {
         int b = kbatch / numNeurons;
         int k = kbatch % numNeurons;
         MEM_GLOBAL pvdata_t * VBatch = V + b*numNeurons;
         MEM_GLOBAL pvdata_t * activityBatch = activity + b*(nx+lt+rt)*(ny+up+dn)*nf;
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (VBatch[k] < VThresh)
            activityBatch[kex] = AMin;
         else if (VBatch[k] < VThresh + VWidth)
            activityBatch[kex] = AMin + (VThresh+VWidth-AShift-AMin)*(VBatch[k]-VThresh)/VWidth;
         else
            activityBatch[kex] -= AShift;
      }
   }
   return PV_SUCCESS;
}
#endif // OBSOLETE // Marked obsolete July 28, 2015.  Use setActivity_PtwiseLinearTransferLayer instead of applyVThresh_ANNLayer and applyVMax_ANNLayer

KERNEL
int applyVThresh_ANNErrorLayer(int nbatch, int numNeurons,
      MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t VThresh,
      pvdata_t AShift, MEM_GLOBAL pvdata_t * activity, int nx, int ny,
      int nf, int lt, int rt, int dn, int up) {
   if (VThresh > -max_pvvdata_t) {
      int kbatch = 0;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
      #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for schedule(static)
      #endif
         for (kbatch = 0; kbatch < numNeurons * nbatch; kbatch++)
#else
         kbatch = getIndex();
#endif // PV_USE_OPENCL
      {
         int b = kbatch / numNeurons;
         int k = kbatch % numNeurons;
         MEM_GLOBAL pvdata_t * VBatch = V + b*numNeurons;
         MEM_GLOBAL pvdata_t * activityBatch = activity + b*(nx+lt+rt)*(ny+up+dn)*nf;
         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         if (fabs(VBatch[k]) < VThresh)
            activityBatch[kex] = AMin;
         else
            activityBatch[kex] -= AShift;
      }
   }
   return PV_SUCCESS;
}

#ifdef OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.
KERNEL
int applyV_ANNLabelLayer(int nbatch, int numNeurons,
                                       MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t AMax, pvdata_t VThresh,MEM_GLOBAL pvdata_t * activity, int nx, int ny,
                                             int nf, int lt, int rt, int dn, int up) {
    if (VThresh > -max_pvvdata_t) {
        int k = 0;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
        for (k = 0; k < numNeurons; k++)
#else
        k = getIndex();
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
#endif // OBSOLETE // Marked obsolete July 28, 2015.  ANNLabelLayer has been moved to obsolete folder.

KERNEL
int squareV_ANNSquaredLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V) {
   int k;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( k=0; k<numNeurons*nbatch; k++ )
#else
      k = getIndex();
#endif // PV_USE_OPENCL
   {
      V[k] *= V[k];
   }
   return PV_SUCCESS;
}

#ifdef OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative
KERNEL
int updateSparsityTermDeriv_GenerativeLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * sparsitytermderivative) {
   int k;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( k=0; k<numNeurons; k++ )
#else
      k = getIndex();
#endif // PV_USE_OPENCL
   {
      pvdata_t vk = V[k];
      sparsitytermderivative[k] = 2*vk/(1+vk*vk);
   }
   return PV_SUCCESS;
}
#endif // OBSOLETE // Marked obsolete July 28, 2015.  GenerativeLayer was moved to inactivesandboxes/SymmetryBreakingGenerative

KERNEL
int updateSparsityTermDeriv_LogLatWTAGenLayer(int nbatch, int numNeurons, int num_features, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * sparsitytermderivative) {
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
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
   int k = getIndex();
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

KERNEL
pvdata_t lateralCompetitionPenalty(MEM_GLOBAL pvdata_t * V, int num_features) {
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

KERNEL
int setActivity_HyPerLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int kbatch;
   #if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
   #else
   kbatch = getIndex();
   #endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      MEM_GLOBAL pvdata_t * ABatch = A + b*((nx+lt+rt)*(ny+up+dn)*nf);
      MEM_GLOBAL pvdata_t * VBatch = V + b*numNeurons;
      int kex = kIndexExtended(k,nx,ny,nf,lt, rt, dn, up);
      ABatch[kex] = VBatch[k];
   }
   return PV_SUCCESS;
}

KERNEL
int setActivity_PtwiseLinearTransferLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, int numVertices, pvdata_t * verticesV, pvdata_t * verticesA, float * slopes) {
   int kbatch;
   int last = numVertices-1;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
#else
      kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      pvdata_t * VBatch = V + b * numNeurons;
      pvdata_t * ABatch = A + b * (nx + lt + rt) * (ny + up + dn) * nf;
      int kex = kIndexExtended(k,nx,ny,nf,lt,rt,dn,up);
      int v;
      pvdata_t potential = VBatch[k];
      pvdata_t activity = 0.0f;
      
      if (potential < verticesV[0]) {
         activity = verticesA[0] + slopes[0]*(potential-verticesV[0]);
      }
      else if (potential >= verticesV[last]) {
         activity = verticesA[last] + slopes[numVertices]*(potential-verticesV[last]);
      }
      else {
         for (v=0; v<last; v++) {
            if (potential<verticesV[v]) { break; } // makes the jumps continuous from the right.  TODO: allow user control over value at jump
            if (potential==verticesV[v]) {
               activity = verticesA[v];
            }
            else if (potential>verticesV[v] && potential<verticesV[v+1]) {
               activity = verticesA[v] + slopes[v+1]*(potential-verticesV[v]);
            }
         }
      }
      ABatch[kex] = activity;
   }
   return PV_SUCCESS;
}

KERNEL
int setActivity_AccumulateLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
//static inline int setActivity_HyPerLayer(int numNeurons, pvdata_t * A, pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int kbatch;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
      #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for schedule(static)
      #endif
   for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
#else
   kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      MEM_GLOBAL pvdata_t * ABatch = A + b*((nx+lt+rt)*(ny+up+dn)*nf);
      MEM_GLOBAL pvdata_t * VBatch = V + b*numNeurons;
      int kex = kIndexExtended(k,nx,ny,nf,lt, rt, dn, up);
      ABatch[kex] += VBatch[k];
   }
   return PV_SUCCESS;
}

KERNEL
int setActivity_GapLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, int orig_lt, int orig_rt, int orig_dn, int orig_up, MEM_GLOBAL pvdata_t * checkActive, float ampSpikelet) {
   int kbatch;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
      #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for schedule(static)
      #endif
      for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
#else
         kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      MEM_GLOBAL pvdata_t* ABatch = A + b*((nx+lt+rt)*(ny+up+dn)*nf);
      MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
      MEM_GLOBAL pvdata_t* checkActiveBatch = checkActive + b*numNeurons;
      int kex = kIndexExtended(k,nx,ny,nf,lt, rt, dn, up);
      int kexorig = kIndexExtended(k,nx,ny,nf,orig_lt, orig_rt, orig_dn, orig_up);
      ABatch[kex] = VBatch[k];
      if( checkActiveBatch[kexorig] > 0.0) ABatch[kex] += ampSpikelet;
   }
   return PV_SUCCESS;
}

KERNEL
int setActivity_SigmoidLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, float VthRest, float Vrest, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag, float dt) {
   pvdata_t Vth = (VthRest+Vrest)/2.0;
   pvdata_t sig_scale = -logf(1.0f/sigmoid_alpha - 1.0f)/(Vth - Vrest);
   if (!sigmoid_flag) {
      sig_scale = sig_scale/logf(3.0f);
      // If sigmoid_flag is off, A is a piecewise linear function of V, with slope of sig_scale/2 at V=Vth, truncated to have minimum value 0 and maximum value 1.
      // The log(3) term makes alpha=1/4 have the slope such that V reaches 0 at Vrest, and V reaches 1 at VthRest.  Without it, that alpha happens at 0.26894...
   }

   int kbatch;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
   for( kbatch=0; kbatch<numNeurons; kbatch++ )
#else
      kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      MEM_GLOBAL pvdata_t* ABatch = A + b*((nx+lt+rt)*(ny+up+dn)*nf);
      MEM_GLOBAL pvdata_t* VBatch = V + b*numNeurons;
      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      pvdata_t activity = 0.0f;
      if(!sigmoid_flag) {
         activity = 0.5f - (VBatch[k] - Vth) * sig_scale/2;
         activity = activity < 0.0f ? 0.0f : activity;
         activity = activity > 1.0f ? 1.0f : activity;
      }
      else{
         activity = 1.0f / (1.0f + exp(2.0f * (VBatch[k] - Vth) * sig_scale));
      }
      ABatch[kex] = activity;
      if (inverse_flag) ABatch[kex] = 1.0f - ABatch[kex];
      // At this point A[kex] is in spikes per milli seconds;
      // A*dt makes activity dimensionless and timestep-independent
      // A[kex] *= dt;
      // This was moved to the strength definition of the dynamic layers

   }
   return PV_SUCCESS;
}

KERNEL
int resetGSynBuffers_HyPerLayer(int nbatch, int numNeurons, int num_channels, MEM_GLOBAL pvdata_t * GSynHead) {
   for( int ch = 0; ch < num_channels; ch ++ ) {
      MEM_GLOBAL pvdata_t * channelStart = &GSynHead[ch*nbatch*numNeurons];
      int k;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
      for( k=0; k<numNeurons*nbatch; k++ )
#else
      k = getIndex();
#endif // PV_USE_OPENCL
      {
         channelStart[k] = 0.0f;
      }
   }
   return PV_SUCCESS;
}

//TODO merge this with resetGSynBuffers_HyPerLayer with a template
KERNEL
int resetGSynBuffers_PoolingIndexLayer(int nbatch, int numNeurons, int num_channels, MEM_GLOBAL int * GSynHead) {
   for( int ch = 0; ch < num_channels; ch ++ ) {
      MEM_GLOBAL int * channelStart = &GSynHead[ch*nbatch*numNeurons];
      int k;
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for schedule(static)
   #endif
      for( k=0; k<numNeurons*nbatch; k++ )
#else
      k = getIndex();
#endif // PV_USE_OPENCL
      {
         channelStart[k] = -1;
      }
   }
   return PV_SUCCESS;
}

KERNEL
int resetGSynBuffers_SigmoidLayer() {
   return PV_SUCCESS; // V is cloned from sourcelayer, so Sigmoid Layer doesn't use the GSynBuffers
}

/* KmeansLayer does not use V */
KERNEL
int setActivity_KmeansLayer(int nbatch, int numNeurons, int num_channels, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * A,int nx,int ny, int nf, int lt, int rt, int dn, int up, bool trainingFlag)
{
    MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
    for(int b = 0; b < nbatch; b++){
       MEM_GLOBAL pvdata_t * ABatch = A + b*((nx+lt+rt)*(ny+up+dn)*nf);
       MEM_GLOBAL pvdata_t * GSynExcBatch = GSynExc + b*numNeurons;
       
       for(int i = 0; i < ny; i++)
       {
           for(int j = 0; j < nx; j++)
           {
               if(trainingFlag)
               {
                   int max = 0,maxIndex = 0;

                   //look for the maximum value
                   for(int k = 0; k < nf; k++)
                   {
                       int kk = (i*nx+j)*nf + k;

                       if(GSynExcBatch[kk] > max)
                       {
                           max = GSynExcBatch[kk];
                           maxIndex = kk;
                       }
                   }
               
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
                   for(int k=0; k<nf; k++ )
                   {
                       int kk = (i*nx+j)*nf + k;
                       int kex = kIndexExtended(kk,nx,ny,nf,lt, rt, dn, up);
                       if (kk == maxIndex)
                           ABatch[kex] = 1;
                       else
                           ABatch[kex] = 0;
                   }
               }
               else
               {
                   //compute mean
                   float mean = 0;
                   
                   for(int k = 0; k < nf; k++)
                   {
                       int kk = (i*nx+j)*nf + k;
                       mean += GSynExcBatch[kk];
                   }

                   mean /= nf;
                   
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
                   for(int  k=0; k<nf; k++ )
                   {
                       int kk = (i*nx+j)*nf + k;
                       int kex = kIndexExtended(kk,nx,ny,nf,lt, rt, dn, up);
                       if (GSynExcBatch[kk] >= mean)
                           ABatch[kex] = GSynExcBatch[kk];
                       else
                           ABatch[kex] = 0;
                   }
               }
           }
       }
    }

    return PV_SUCCESS;
}

KERNEL
int setActivity_MLPSigmoidLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, float linear_alpha, bool* dropout_buf, int nx, int ny, int nf, int lt, int rt, int dn, int up, float dt) {
   int kbatch;
   
#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
   #ifdef PV_USE_OPENMP_THREADS
   #pragma omp parallel for
   #endif
   for( kbatch=0; kbatch<numNeurons*nbatch; kbatch++ )
#else
      kbatch = getIndex();
#endif // PV_USE_OPENCL
   {
      int b = kbatch / numNeurons;
      int k = kbatch % numNeurons;
      pvdata_t * VBatch = V + b * nx * ny * nf;
      pvdata_t * ABatch = A + b * (nx + lt + rt) * (ny + up + dn) * nf;
      bool* dropoutBatch = dropout_buf + b * nx * ny * nf;

      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      pvdata_t activity = 1.7159 * tanh(((float)2/3) * VBatch[k]) + linear_alpha * VBatch[k];
      //Set explicitly to 0 if that neuron was dropped out
      ABatch[kex] = dropoutBatch[k] ? 0 : activity;
   }
   return PV_SUCCESS;
}

//Not used, deprecated 6/17
//KERNEL
//int calcGSyn_Mean_StdDev(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * GSynHead,
//      MEM_GLOBAL double * GSyn_Mean, MEM_GLOBAL double * GSyn_StdDev) {
//   int k;
//   MEM_GLOBAL pvdata_t * GSynError = &GSynHead[0 * nbatch * numNeurons]; // weighted input
//   *GSyn_Mean = 0;
//   *GSyn_StdDev = 0;
//#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
//   #ifdef PV_USE_OPENMP_THREADS
//   #pragma omp parallel for
//   #endif
//   for (k = 0; k < numNeurons; k++)
//#else
//   k = getIndex();
//#endif // PV_USE_OPENCL
//   {
//      *GSyn_Mean += GSynError[k];
//      *GSyn_StdDev += GSynError[k] * GSynError[k];
//   }
//   return PV_SUCCESS;
//}

//KERNEL
//int updateV_ANNLabelLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V,
//      MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL float * activity, pvdata_t AMax,
//      pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, int nx, int ny, int nf, int lt, int rt, int dn, int up)
//{
//   int status;
//   status = applyGSyn_HyPerLayer(nbatch, numNeurons, V, GSynHead);
//   if(status == PV_SUCCESS) status = setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);
//   if( status == PV_SUCCESS ) status =
//                                  applyV_ANNLabelLayer(nbatch, numNeurons, V, AMin, AMax, VThresh, activity, nx, ny, nf, lt, rt, dn, up);
//
//   return status;
//}

//KERNEL int updateV_TrainingLayer(int nbatch, int numNeurons,
//      MEM_GLOBAL pvdata_t * V, int numTrainingLabels, int * trainingLabels,
//      int traininglabelindex, int strength) {
//   if(traininglabelindex<0 || traininglabelindex>=numTrainingLabels){
//      return PV_FAILURE;
//   }
//   int n = trainingLabels[traininglabelindex];
//   if(n<0 || n>=numNeurons){
//      return PV_FAILURE;
//   }
//   V[trainingLabels[traininglabelindex]] = 0;
//   traininglabelindex++;
//   if (traininglabelindex == numTrainingLabels)
//      traininglabelindex = 0;
//   if (trainingLabels[traininglabelindex] >= 0)
//      V[trainingLabels[traininglabelindex]] = strength;
//   return PV_SUCCESS;
//}

//KERNEL
//int update_dV_GenerativeLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * GSynHead, MEM_GLOBAL pvdata_t * sparsitytermderivative, MEM_GLOBAL  pvdata_t * dV, pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence) {
//   int k;
//   MEM_GLOBAL pvdata_t * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
//   MEM_GLOBAL pvdata_t * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
//   MEM_GLOBAL pvdata_t * GSynAux = &GSynHead[CHANNEL_INHB*nbatch*numNeurons];
//#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
//   #ifdef PV_USE_OPENMP_THREADS
//   #pragma omp parallel for
//   #endif
//   for( k=0; k<numNeurons*nbatch; k++ )
//#else
//      k = getIndex();
//#endif // PV_USE_OPENCL
//   {
//      //TODO sparsitytermderivative
//      pvdata_t dAnew = GSynExc[k] - GSynInh[k] + auxChannelCoeff*GSynAux[k] - sparsityTermCoeff*sparsitytermderivative[k];
//      dV[k] = persistence*dV[k] + (1-persistence)*dAnew;
//   }
//   return PV_SUCCESS;
//}

//KERNEL
//int applyV_ANNLabelLayer(int numNeurons,
//                                       MEM_GLOBAL pvdata_t * V, pvdata_t AMin, pvdata_t AMax, pvdata_t VThresh,MEM_GLOBAL pvdata_t * activity, int nx, int ny,
//                                             int nf, int lt, int rt, int dn, int up) {
//    if (VThresh > -max_pvvdata_t) {
//        int k = 0;
//#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
//   #ifdef PV_USE_OPENMP_THREADS
//   #pragma omp parallel for
//   #endif
//        for (k = 0; k < numNeurons; k++)
//#else
//        k = getIndex();
//#endif // PV_USE_OPENCL
//        {
//            int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
//            int featureindex = featureIndex(k, nx, ny, nf) % nf;
//            float factor;
//            if (nf == 2)
//                factor = 1.0;
//            else
//                factor = 255.0;
//            
//            if (fabs(fabs(V[k]) * factor - featureindex) < 0.00001)
//                activity[kex] = AMax;
//            else
//                activity[kex] = AMin;
//        }
//    }
//    return PV_SUCCESS;
//}
//KERNEL
//int updateSparsityTermDeriv_GenerativeLayer(int numNeurons, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * sparsitytermderivative) {
//   int k;
//#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
//   #ifdef PV_USE_OPENMP_THREADS
//   #pragma omp parallel for
//   #endif
//   for( k=0; k<numNeurons; k++ )
//#else
//      k = getIndex();
//#endif // PV_USE_OPENCL
//   {
//      pvdata_t vk = V[k];
//      sparsitytermderivative[k] = 2*vk/(1+vk*vk);
//   }
//   return PV_SUCCESS;
//}

//KERNEL
//int updateSparsityTermDeriv_LogLatWTAGenLayer(int numNeurons, int num_features, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * sparsitytermderivative) {
//#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
//   int k;
//   #ifdef PV_USE_OPENMP_THREADS
//   #pragma omp parallel for
//   #endif
//   for (k=0; k<numNeurons/num_features; k++) {
//      int feature_start = k*num_features;
//      pvdata_t sum_across_features = 0.0f;
//      int f;
//      for (f=0; f<num_features; f++) {
//         sum_across_features += V[feature_start+f];
//      }
//      pvdata_t lat_wta_expr = lateralCompetitionPenalty(&V[feature_start], num_features);
//      for (f=0; f<num_features; f++) {
//         sparsitytermderivative[k*num_features+f] = 2*(sum_across_features-V[k*num_features+f])/(1+lat_wta_expr);
//      }
//   }
//   for (k=0; k<numNeurons; k++) {
//
//   }
//#else // PV_USE_OPENCL
//   int k = getIndex();
//   {
//      int feature_start = k - (k % num_features);
//      pvdata_t sum_across_features = 0.0f;
//      for( int f=0; f<num_features; f++ ) sum_across_features += V[feature_start+f];
//      pvdata_t lat_wta_expr = lateralCompetitionPenalty(&V[feature_start], num_features);
//      // Each block of num_features neurons will have the same sum_across_features and latWTAexpr.
//      // Can we eliminate redundant calculations?
//      sparsitytermderivative[k] = 2*(sum_across_features-V[k])/(1+lat_wta_expr);
//   }
//#endif // PV_USE_OPENCL
//
//   return PV_SUCCESS;
//}
//KERNEL
//pvdata_t lateralCompetitionPenalty(MEM_GLOBAL pvdata_t * V, int num_features) {
//   pvdata_t z=0;
//   #ifdef PV_USE_OPENMP_THREADS
//   #pragma omp parallel for
//   #endif
//   for( int p=0; p<num_features; p++ ) {
//      for( int q=0; q<num_features; q++ ) {
//         if( p!= q ) z += V[p]*V[q];
//      }
//   }
//   return z;
//}

//KERNEL
//int setActivity_GenerativeLayer(int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, int nx, int ny, int nf, int lt, int rt, int dn, int up, pvdata_t activity_threshold) {
//   int k;
//#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
//   #ifdef PV_USE_OPENMP_THREADS
//   #pragma omp parallel for
//   #endif
//   for( k=0; k<numNeurons; k++ )
//#else
//      k = getIndex();
//#endif // PV_USE_OPENCL
//   {
//      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
//      A[kex] = fabs(V[k])>activity_threshold ? V[k] : 0.0f;
//   }
//   return PV_SUCCESS;
//}

//KERNEL
//int setActivity_IncrementLayer(int nbatch, int numNeurons, MEM_GLOBAL pvdata_t * A, MEM_GLOBAL pvdata_t * V, MEM_GLOBAL pvdata_t * Vprev, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
//   int k;
//
//   for(int b = 0; b < nbatch; b++){
//      MEM_GLOBAL pvdata_t * ABatch = A + b*((nx+lt+rt)*(ny+up+dn)*nf);
//      MEM_GLOBAL pvdata_t * VBatch = V + b*numNeurons;
//      MEM_GLOBAL pvdata_t * VprevBatch = Vprev + b*numNeurons;
//
//#if !defined(PV_USE_OPENCL) && !defined(PV_USE_CUDA)
//      #ifdef PV_USE_OPENMP_THREADS
//      #pragma omp parallel for
//      #endif
//      for( k=0; k<numNeurons; k++ )
//#else
//         k = getIndex();
//#endif // PV_USE_OPENCL
//      {
//         int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
//         ABatch[kex] = VBatch[k]-VprevBatch[k];
//      }
//   }
//   return PV_SUCCESS;
//}


#endif /* UPDATESTATEFUNCTIONS_H_ */
