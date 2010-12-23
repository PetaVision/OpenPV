#ifndef PV_USE_OPENCL

#include "../../arch/opencl/pv_opencl.h"

#else

// This stuff needs to be moved to (or obtained from) separate file
// perhaps conversions.h

#define pvdata_t float
#define PHI_EXC 0       // which phi buffer to use
#define PHI_INH 1       // which phi buffer to use
#define PHI_INHB 2
#define PI              3.1415926535897932

#define PV_RANDOM_MAX       0x7fffffff
#define PV_INV_RANDOM_MAX   (1.0 / (double) PV_RANDOM_MAX)

static inline double pv_random_prob()
{
//   return (double) random() * PV_INV_RANDOM_MAX;
   return 1.0;
}

static inline int featureIndex(int k, int nx, int ny, int nf)
{
   return k % nf;
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

#endif // PV_USE_OPENCL


int spike(float time, float dt, float prev, float probBase, float probStim, float * probSpike)
{
//   fileread_params * params = (fileread_params *) clayer->params;
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
      float refact = 1.0f - expf(-delta/REFACTORY_PERIOD);
      refact = (refact < 0) ? 0 : refact;
      probBase *= refact;
      probStim *= refact;
   }

   if (burstDuration <= 0 || burstFreq == 0) {
      sinAmp = cos( 2 * PI * time * burstFreq / 1000. );
   }
   else {
      burstStatus = fmodf(time, 1000. / burstFreq);
      burstStatus = burstStatus <= burstDuration;
   }

   burstStatus *= (int) ( (time >= beginStim) && (time < endStim) );
   *probSpike = probBase;

   if ((int)burstStatus) {
      *probSpike += probStim * sinAmp;  // negative prob is OK
    }
   return ( pv_random_prob() < *probSpike );
}

//
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void Retina_update_state (
    const float time,
    const float dt,
    const float probStim,
    const float probBase,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL float * phi,
    CL_MEM_GLOBAL float * activity,
    CL_MEM_GLOBAL float * prevActivity)
{
#ifndef PV_USE_OPENCL
for (int k = 0; k < nx*ny*nf; k++) {
#else   
   int k = get_global_id(0);
#endif

   int kx = kxPos(k, nx, ny, nf);
   int ky = kyPos(k, nx, ny, nf);

   //
   // kernel (nonheader) begins here
   //

   float probSpike;
   const int numNeurons = nx * ny * nf;
   const int kex = kIndexExtended(k, nx, ny, nf, nb);

   CL_MEM_GLOBAL pvdata_t * phiExc = & phi[PHI_EXC * numNeurons];
   CL_MEM_GLOBAL pvdata_t * phiInh = & phi[PHI_INH * numNeurons];

   const float prevTime = prevActivity[kex];
   const float l_V = phiExc[k] - phiInh[k]; 

   phiExc[k] = 0.0f;
   phiInh[k] = 0.0f;

   const float l_activity = spike(time, dt, prevTime, probBase, probStim*l_V, &probSpike);
   activity[kex] = l_activity;
   prevActivity[kex] = (l_activity > 0.0f) ? time : prevTime;

#ifndef PV_USE_OPENCL
   }
#endif

}
