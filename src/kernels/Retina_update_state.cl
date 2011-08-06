#include "Retina_params.h"
#include "cl_random.hcl"

#ifndef PI
#  define PI 3.1415926535897932
#endif

#ifndef PV_USE_OPENCL

#  include <math.h>
#  define EXP  expf
#  define COS  cosf
#  define FMOD fmodf
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_LOCAL

#else

#  define EXP exp
#  define COS  __cosf
#  define FMOD __fmodf
#  define CL_KERNEL     __kernel
#  define CL_MEM_GLOBAL __global
#  define CL_MEM_LOCAL  __local
#  include "conversions.hcl"

#endif /* PV_USE_OPENCL */

/*
 * Spiking method for Retina
 * Returns 1 if an event should occur, 0 otherwise. This is a stochastic model.
 *
 * REMARKS:
 *      - During ABS_REFACTORY_PERIOD a neuron does not spike
 *      - The neurons that correspond to stimuli (on Image pixels)
 *        spike with probability probStim.
 *      - The neurons that correspond to background image pixels
 *        spike with probability probBase.
 *      - After ABS_REFACTORY_PERIOD the spiking probability
 *        grows exponentially to probBase and probStim respectively.
 *      - The burst of the retina is periodic with period T set by
 *        T = 1000/burstFreq in miliseconds
 *      - When the time t is such that mT < t < mT + burstDuration, where m is
 *        an integer, the burstStatus is set to 1.
 *      - The burstStatus is also determined by the condition that
 *        beginStim < t < endStim. These parameters are set in the input
 *        params file params.stdp
 *      - sinAmp modulates the spiking probability only when burstDuration <= 0
 *        or burstFreq = 0
 *      - probSpike is set to probBase for all neurons.
 *      - for neurons exposed to Image on pixels, probSpike increases with probStim.
 *      - When the probability is negative, the neuron does not spike.
 *
 * NOTES:
 *      - time is measured in milliseconds.
 *
 */
static inline 
int spike(float time, float dt,
          float prev, float stimFactor, uint4 * rnd_state, CL_MEM_GLOBAL Retina_params * params)
{
   float probSpike;
   float burstStatus = 1;
   float sinAmp = 1.0;
   
   // input parameters
   //
   float probBase  = params->probBase;
   float probStim  = params->probStim * stimFactor;

   // see if neuron is in a refactory period
   //
   if ((time - prev) < params->abs_refactory_period) {
      return 0;
   }
   else {
      float delta = time - prev - params->abs_refactory_period;
      float refact = 1.0f - EXP(-delta/params->refactory_period);
      refact = (refact < 0) ? 0 : refact;
      probBase *= refact;
      probStim *= refact;
   }

   if (params->burstDuration <= 0 || params->burstFreq == 0) {
      sinAmp = COS( 2*PI*time * params->burstFreq / 1000. );
   }
   else {
      burstStatus = FMOD(time, 1000. / params->burstFreq);
      burstStatus = burstStatus <= params->burstDuration;
   }

   burstStatus *= (int) ( (time >= params->beginStim) && (time < params->endStim) );
   probSpike = probBase;

   if ((int)burstStatus) {
      probSpike += probStim * sinAmp;  // negative prob is OK
   }

   *rnd_state = cl_random_get(*rnd_state);
   int spike_flag = (cl_random_prob(*rnd_state) < probSpike);
   return spike_flag;
}

//
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void Retina_spiking_update_state (
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL Retina_params * params,
    CL_MEM_GLOBAL uint4 * rnd,
    CL_MEM_GLOBAL float * phiExc,
    CL_MEM_GLOBAL float * phiInh,
    CL_MEM_GLOBAL float * activity,
    CL_MEM_GLOBAL float * prevTime)
{
   int k;
#ifndef PV_USE_OPENCL
for (k = 0; k < nx*ny*nf; k++) {
#else   
   k = get_global_id(0);
#endif

   int kex = kIndexExtended(k, nx, ny, nf, nb);

   //
   // kernel (nonheader part) begins here
   //
   
   // load local variables from global memory
   //
   uint4 l_rnd = rnd[k]; 
   float l_phiExc = phiExc[k];
   float l_phiInh = phiInh[k];
   float l_prev   = prevTime[kex];
   float l_activ;

   l_activ = (float) spike(time, dt, l_prev, (l_phiExc - l_phiInh), &l_rnd, params);
   l_prev  = (l_activ > 0.0f) ? time : l_prev;

   l_phiExc = 0.0f;
   l_phiInh = 0.0f;

   // store local variables back to global memory
   //
   rnd[k] = l_rnd;
   phiExc[k] = l_phiExc;
   phiInh[k] = l_phiInh;
   prevTime[kex] = l_prev;
   activity[kex] = l_activ;

#ifndef PV_USE_OPENCL
   }
#endif

}

//
// update the state of a retinal layer (non-spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void Retina_nonspiking_update_state (
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL Retina_params * params,
    CL_MEM_GLOBAL float * phiExc,
    CL_MEM_GLOBAL float * phiInh,
    CL_MEM_GLOBAL float * activity)
{
   int k;
#ifndef PV_USE_OPENCL
for (k = 0; k < nx*ny*nf; k++) {
#else   
   k = get_global_id(0);
#endif

   int kex = kIndexExtended(k, nx, ny, nf, nb);

   //
   // kernel (nonheader part) begins here
   //
   
   // load local variables from global memory
   //
   float l_phiExc = phiExc[k];
   float l_phiInh = phiInh[k];
   float l_activ;

   // adding base prob should not change default behavior
   l_activ = params->probStim*(l_phiExc - l_phiInh) + params->probBase;

   l_phiExc = 0.0f;
   l_phiInh = 0.0f;

   // store local variables back to global memory
   //
   phiExc[k] = l_phiExc;
   phiInh[k] = l_phiInh;
   activity[kex] = l_activ;

#ifndef PV_USE_OPENCL
   }
#endif

}
