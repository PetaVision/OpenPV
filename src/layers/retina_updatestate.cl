//#include <OpenCL/opencl.h>

#ifdef DEBUG_PRINT
#  include <stdio.h>
#endif

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

// problems including stdlib.h
double random();

static inline double pv_random_prob()
{
   return (double) random() * PV_INV_RANDOM_MAX;
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

static inline int kIndexRestricted(int k_ex, int nx, int ny, int nf, int nb)
{
   int kx, ky, kf;

   const int nx_ex = nx + 2*nb;
   const int ny_ex = ny + 2*nb;

   kx = kxPos(k_ex, nx_ex, ny_ex, nf) - nb;
   if (kx < 0 || kx >= nx) return -1;

   ky = kyPos(k_ex, nx_ex, ny_ex, nf) - nb;
   if (ky < 0 || ky >= ny) return -1;

   kf = featureIndex(k_ex, nx_ex, ny_ex, nf);
   return kIndex(kx, ky, kf, nx, ny, nf);
}

inline uint k_index(uint nx, uint ny)
{
   return (get_global_id(0) + nx * get_global_id(1));
}

/**
 * Returns the extended index
 *  - assumes kernel called with extended thread space
 */
inline uint k_ex_index(uint nx, uint ny, uint nPad)
{
   return (get_global_id(0) + (nx+2*nPad) * get_global_id(1));
}

inline uint kx_index()
{
   return get_global_id(0);
}

inline uint ky_index()
{
   return get_global_id(1);
}

#undef SIMPLE_SPIKE
#ifdef SIMPLE_SPIKE
int spike(float time, float dt, float prev, float probBase, float probStim, float * probSpike)
{
   return 1;
}
#else
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
#endif

//
// update the state of a retinal layer (spiking)
//
// This will be called with grid layout in extended space (nx*2*nPad)*(ny*2*nPad)
//
__kernel void update_state (
    const float time,
    const float dt,
    const float probStim,
    const float probBase,

    const uint nx,
    const uint ny,
    const uint nf,
    const uint nPad,

    __global float * phi,
    __global float * activity,
    __global float * prevActivity)
{
   //
   // Make sure threads are within bounds.
   //
   if (kx_index() >= nx+2*nPad || ky_index() >= ny+2*nPad) {
      return;
   }

   float probSpike;
   const uint numNeurons = nx * ny * nf;

   __global pvdata_t * phiExc = & phi[PHI_EXC * numNeurons];
   __global pvdata_t * phiInh = & phi[PHI_INH * numNeurons];

   const uint kex = k_ex_index(nx, ny, nPad);
   const int k = kIndexRestricted(kex, nx, ny, nf, nPad);
      
   const float prevTime = prevActivity[kex];
   
   if (k >= 0) {
      //
      // interior region
      //
      const float l_V = phiExc[k] - phiInh[k]; 

      phiExc[k] = 0.0f;
      phiInh[k] = 0.0f;

      const float l_activity = spike(time, dt, prevTime, probBase, probStim*l_V, &probSpike);
      activity[kex] = l_activity;
      prevActivity[kex] = (l_activity > 0.0f) ? time : prevTime;

#ifdef DEBUG_PRINT
      if (k < 6) {
         printf("k==%d kex==%d kx==%d ky==%d l_V==%f\n", k, kex, kx_index(), ky_index(), l_V);
      }
#endif
   }
   else {
      //
      // border region (background activity only)
      //
      const float l_activity = spike(time, dt, prevTime, probBase, 0.0f, &probSpike);
      activity[kex] = l_activity;
      prevActivity[kex] = (l_activity > 0.0f) ? time : prevTime;
   }
}
