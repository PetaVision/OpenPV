#include "Retina_params.h"
#include "cl_random.hcl"

#ifndef PI
#  define PI 3.1415926535897932
#endif

#ifndef PV_USE_OPENCL

#  include <math.h>
#  define EXP  expf
#  define COS  cosf
#  define MOD fmod
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL

#else

#  define EXP exp
#  define COS  __cosf
#  define FMOD __fmodf
#  define CL_KERNEL     __kernel
#  define CL_MEM_GLOBAL __global
#  define CL_MEM_CONST    __constant
#  define CL_MEM_LOCAL  __local
#  include "conversions.hcl"
#  define CHANNEL_EXC   0
#  define CHANNEL_INH   1
#  define CHANNEL_INHB  2
#  define CHANNEL_GAP   3

#endif /* PV_USE_OPENCL */

/*
 * Spiking method for Retina
 * Returns 1 if an event should occur, 0 otherwise. This is a stochastic model.
 *
 * REMARKS:
 *      - During ABS_REFRACTORY_PERIOD a neuron does not spike
 *      - The neurons that correspond to stimuli (on Image pixels)
 *        spike with probability probStim.
 *      - The neurons that correspond to background image pixels
 *        spike with probability probBase.
 *      - After ABS_REFRACTORY_PERIOD the spiking probability
 *        grows exponentially to probBase and probStim respectively.
 *      - The burst of the retina is periodic with period T set by
 *        T = 1000/burstFreq in milliseconds
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
float calcBurstStatus(double timed, CL_MEM_CONST Retina_params * params) {
   float burstStatus;
   if (params->burstDuration <= 0 || params->burstFreq == 0) {
      burstStatus = COS( 2*PI*timed * params->burstFreq / 1000. );
   }
   else {
      burstStatus = MOD(timed, 1000. / params->burstFreq);
      burstStatus = burstStatus < params->burstDuration;
   }
   burstStatus *= (int) ( (timed >= params->beginStim) && (timed < params->endStim) );
   return burstStatus;
}

static inline
int spike(float timed, float dt,
          float prev, float stimFactor, taus_uint4 * rnd_state, float burst_status, CL_MEM_CONST Retina_params * params)
{
   float probSpike;

   // input parameters
   //
   float probBase  = params->probBase;
   float probStim  = params->probStim * stimFactor;

   // see if neuron is in a refractory period
   //
   if ((timed - prev) < params->abs_refractory_period) {
      return 0;
   }
   else {
      float delta = timed - prev - params->abs_refractory_period;
      float refract = 1.0f - EXP(-delta/params->refractory_period);
      refract = (refract < 0) ? 0 : refract;
      probBase *= refract;
      probStim *= refract;
   }

   probSpike = probBase;

   probSpike += probStim * burst_status;  // negative prob is OK
   // probSpike is spikes per millisecond; conversion to expected number of spikes in dt takes place in setRetinaParams

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
    const int nbatch,
    const int numNeurons,
    const double timed,
    const double dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    CL_MEM_CONST Retina_params * params,
    CL_MEM_GLOBAL taus_uint4 * rnd,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity,
    CL_MEM_GLOBAL float * prevTime)
{

   CL_MEM_GLOBAL float * phiExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   CL_MEM_GLOBAL float * phiInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
   for(int b = 0; b < nbatch; b++){
      CL_MEM_GLOBAL taus_uint4* rndBatch = rnd + b * nx*ny*nf;
      CL_MEM_GLOBAL float* phiExcBatch = phiExc + b*nx*ny*nf;
      CL_MEM_GLOBAL float* phiInhBatch = phiInh + b*nx*ny*nf;
      CL_MEM_GLOBAL float* prevTimeBatch = prevTime + b * (nx+lt+rt)*(ny+up+dn)*nf;
      CL_MEM_GLOBAL float* activityBatch = activity + b * (nx+lt+rt)*(ny+up+dn)*nf;

      
      int k;
      float burst_status = calcBurstStatus(timed, params);
   #ifndef PV_USE_OPENCL
   for (k = 0; k < nx*ny*nf; k++) {
   #else
      k = get_global_id(0);
   #endif

      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);

      //
      // kernel (nonheader part) begins here
      //
      
      // load local variables from global memory
      //
      taus_uint4 l_rnd = rndBatch[k]; 
      float l_phiExc = phiExcBatch[k];
      float l_phiInh = phiInhBatch[k];
      float l_prev   = prevTimeBatch[kex];
      float l_activ;

      l_activ = (float) spike(timed, dt, l_prev, (l_phiExc - l_phiInh), &l_rnd, burst_status, params);
      l_prev  = (l_activ > 0.0f) ? timed : l_prev;

      //l_phiExc = 0.0f;
      //l_phiInh = 0.0f;

      // store local variables back to global memory
      //
      rndBatch[k] = l_rnd;
      prevTimeBatch[kex] = l_prev;
      activityBatch[kex] = l_activ;
#ifndef PV_USE_OPENCL
      }
#endif
   }

}

//
// update the state of a retinal layer (non-spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void Retina_nonspiking_update_state (
    const int nbatch,
    const int numNeurons,
    const double timed,
    const double dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    CL_MEM_CONST Retina_params * params,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity)
{
   int k;
   float burstStatus = calcBurstStatus(timed, params);

   CL_MEM_GLOBAL float * phiExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   CL_MEM_GLOBAL float * phiInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];

   for(int b = 0; b < nbatch; b++){
      CL_MEM_GLOBAL float* phiExcBatch = phiExc + b*nx*ny*nf;
      CL_MEM_GLOBAL float* phiInhBatch = phiInh + b*nx*ny*nf;
      CL_MEM_GLOBAL float* activityBatch = activity + b * (nx+lt+rt)*(ny+up+dn)*nf;

   #ifndef PV_USE_OPENCL
   for (k = 0; k < nx*ny*nf; k++) {
   #else
      k = get_global_id(0);
   #endif

      int kex = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);

      //
      // kernel (nonheader part) begins here
      //
      
      // load local variables from global memory
      //
      float l_phiExc = phiExcBatch[k];
      float l_phiInh = phiInhBatch[k];
      float l_activ;

      // adding base prob should not change default behavior
      l_activ = burstStatus * params->probStim*(l_phiExc - l_phiInh) + params->probBase;

      //l_phiExc = 0.0f;
      //l_phiInh = 0.0f;

      // store local variables back to global memory
      //
      activityBatch[kex] = l_activ;

   #ifndef PV_USE_OPENCL
      }
   #endif
   }

}
