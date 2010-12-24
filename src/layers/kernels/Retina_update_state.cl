#ifndef PV_USE_OPENCL

#include "../../arch/opencl/pv_opencl.h"

#define EXP expf
#define FMOD fmodf

#else

#define CL_KERNEL     __kernel
#define CL_MEM_GLOBAL __global
#define CL_MEM_LOCAL  __local

#define EXP exp
#define FMOD __fmodf

// This stuff needs to be moved to (or obtained from) separate file
// perhaps conversions.h

#define pvdata_t float
#define PHI_EXC 0       // which phi buffer to use
#define PHI_INH 1       // which phi buffer to use
#define PHI_INHB 2
#define PI              3.1415926535897932

// refactory period for neurons (retina for now)
#define ABS_REFACTORY_PERIOD 3
#define REFACTORY_PERIOD     5

#define PV_RANDOM_MAX       0x7fffffff
#define PV_INV_RANDOM_MAX   (1.0 / (double) PV_RANDOM_MAX)

static inline double pv_random_prob()
{
//   return (double) random() * PV_INV_RANDOM_MAX;
   return 1.0;
}

static inline int kxPos(int k, int nx, int ny, int nf)
{
   return (k/nf) % nx;
}

static inline int kyPos(int k, int nx, int ny, int nf)
{
   return k / (nx*nf);
}

static inline int kIndex(int kx, int ky, int kf, int nx, int ny, int nf)
{
   return kf + (kx + (ky * nx)) * nf;
}

static inline int featureIndex(int k, int nx, int ny, int nf)
{
   return k % nf;
}

static inline int kIndexExtended(int k, int nx, int ny, int nf, int nb)
{
   const int kx_ex = nb + kxPos(k, nx, ny, nf);
   const int ky_ex = nb + kyPos(k, nx, ny, nf);
   const int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + 2*nb, ny + 2*nb, nf);
}

#endif // PV_USE_OPENCL


static int spike(float time, float dt, float prev, float probBase, float probStim)
{
   float probSpike;
   float burstStatus = 1;
   float sinAmp = 1.0;
   
   // input parameters
   // TODO - how to get these into kernel
   //
   const float beginStim = 0.0f;
   const float endStim   = 9000000.0f;
   const float burstFreq = 40.0f;
   const float burstDuration = 7.5f;

   // see if neuron is in a refactory period
   //
   if ((time - prev) < ABS_REFACTORY_PERIOD) {
      return 0;
   }
   else {
      float delta = time - prev - ABS_REFACTORY_PERIOD;
      float refact = 1.0f - EXP(-delta/REFACTORY_PERIOD);
      refact = (refact < 0) ? 0 : refact;
      probBase *= refact;
      probStim *= refact;
   }

   if (burstDuration <= 0 || burstFreq == 0) {
      sinAmp = cos( 2 * PI * time * burstFreq / 1000. );
   }
   else {
      burstStatus = FMOD(time, 1000. / burstFreq);
      burstStatus = burstStatus <= burstDuration;
   }

   burstStatus *= (int) ( (time >= beginStim) && (time < endStim) );
   probSpike = probBase;

   if ((int)burstStatus) {
      probSpike += probStim * sinAmp;  // negative prob is OK
   }
   return ( pv_random_prob() < probSpike );
}

//
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void update_state (
    const float time,
    const float dt,
    const float probStim,
    const float probBase,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL float * phiExc,
    CL_MEM_GLOBAL float * phiInh,
    CL_MEM_GLOBAL float * activity,
    CL_MEM_GLOBAL float * prevTime)
{
#ifndef PV_USE_OPENCL
for (register int k = 0; k < nx*ny*nf; k++) {
#else   
   register int k = get_global_id(0);
#endif

   register int kex = kIndexExtended(k, nx, ny, nf, nb);

   //
   // kernel (nonheader part) begins here
   //

   // load local copies
   //
   register float l_phiExc = phiExc[k];
   register float l_phiInh = phiInh[k];
   register float l_prev   = prevTime[kex];
   register float l_activ  = activity[kex];

   l_activ = spike(time, dt, l_prev, probBase, (l_phiExc - l_phiInh)*probStim);
   l_prev  = (l_activ > 0.0f) ? time : l_prev;

   l_phiExc = 0.0f;
   l_phiInh = 0.0f;

   // store local copies
   //
   phiExc[k] = l_phiExc;
   phiInh[k] = l_phiInh;
   prevTime[kex] = l_prev;
   activity[kex] = l_activ;

#ifndef PV_USE_OPENCL
   }
#endif

}
