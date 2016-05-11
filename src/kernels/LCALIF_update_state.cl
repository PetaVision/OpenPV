#include "LIF_params.h"
#include "cl_random.hcl"
#include "../include/pv_datatypes.h"
#include <float.h>
#include <stdbool.h>

#ifndef PV_USE_OPENCL
#  include <math.h>
#  define EXP expf
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL
#else  /* compiling with OpenCL */
#  define EXP exp
#  define CL_KERNEL       __kernel
#  define CL_MEM_GLOBAL   __global
#  define CL_MEM_CONST    __constant
#  define CL_MEM_LOCAL    __local
#  include "conversions.hcl"
#  define CHANNEL_EXC   0
#  define CHANNEL_INH   1
#  define CHANNEL_INHB  2
#  define CHANNEL_GAP   3
#endif

inline
float LCALIF_tauInf(const float tau, const float G_E, const float G_I, const float G_IB, const pvgsyndata_t sum_gap) {
   return tau/(1+G_E+G_I+G_IB+sum_gap);
}

inline
float LCALIF_VmemInf(const float Vrest, const float V_E, const float V_I, const float V_B, const float G_E, const float G_I, const float G_B, const float G_gap, const pvgsyndata_t sumgap) {
   return (Vrest + V_E*G_E + V_I*G_I + V_B*G_B + G_gap)/(1+G_E+G_I+G_B+sumgap);
}

//
// update the state of a LCALIF layer (spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void LCALIF_update_state(
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
    
    float Vscale,
    float * Vadpt,
    float tauTHR,
    const float targetRateHz,

    float * integratedSpikeCount,
    
    CL_MEM_CONST LIF_params * params,
    CL_MEM_GLOBAL taus_uint4 * rnd,
    CL_MEM_GLOBAL float * V,
    CL_MEM_GLOBAL float * Vth,
    CL_MEM_GLOBAL float * G_E,
    CL_MEM_GLOBAL float * G_I,
    CL_MEM_GLOBAL float * G_IB,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity, 

    const pvgsyndata_t * gapStrength,
    CL_MEM_GLOBAL float * Vattained,
    CL_MEM_GLOBAL float * Vmeminf,
    const int normalizeInputFlag,
    CL_MEM_GLOBAL float * GSynExcEffective,
    CL_MEM_GLOBAL float * GSynInhEffective,
    CL_MEM_GLOBAL float * excitatoryNoise,
    CL_MEM_GLOBAL float * inhibitoryNoise,
    CL_MEM_GLOBAL float * inhibNoiseB)
{

   // convert target rate from Hz to kHz
   float targetRatekHz = targetRateHz/1000;

   // tau parameters
   const float tauO = 1/targetRatekHz;   //Convert target rate from kHz to ms (tauO)

   const float decayE   = EXP(-dt/params->tauE);
   const float decayI   = EXP(-dt/params->tauI);
   const float decayIB  = EXP(-dt/params->tauIB);
   const float decayVth = EXP(-dt/params->tauVth);
   const float decayO   = EXP(-dt/tauO);

   //Convert dt to seconds
   const float dt_sec = .001 * dt;

#ifndef PV_USE_OPENCL
for (int k = 0; k < nx*ny*nf*nbatch; k++) {
#else   
   k = get_global_id(0);
#endif

   int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

   //
   // kernel (nonheader part) begins here
   //

   // local param variables
   float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

   // const float GMAX = 10.0;

   // local variables
   float l_activ;

   taus_uint4 l_rnd = rnd[k];

   float l_V   = V[k];
   float l_Vth = Vth[k];

   // The correction factors to the conductances are so that if l_GSyn_* is the same every timestep,
   // then the asymptotic value of l_G_* will be l_GSyn_*
   float l_G_E  = G_E[k];
   float l_G_I  = G_I[k];
   float l_G_IB = G_IB[k];
   pvgsyndata_t l_gapStrength = gapStrength[k];

#define CHANNEL_NORM (CHANNEL_GAP+1)
   CL_MEM_GLOBAL float * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynInhB = &GSynHead[CHANNEL_INHB*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynGap = &GSynHead[CHANNEL_GAP*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynNorm = &GSynHead[CHANNEL_NORM*nbatch*numNeurons];
   float l_GSynExc  = GSynExc[k];
   float l_GSynInh  = GSynInh[k];
   float l_GSynInhB = GSynInhB[k];
   float l_GSynGap  = GSynGap[k];
   float l_GSynNorm = normalizeInputFlag ? GSynNorm[k] : 1.0f;
   
   // define local param variables
   //
   tau        = params->tau;
   Vexc       = params->Vexc;
   Vinh       = params->Vinh;
   VinhB      = params->VinhB;
   Vrest      = params->Vrest;

   VthRest  = params->VthRest;
   deltaVth = params->deltaVth;
   deltaGIB = params->deltaGIB;

   // TODO OpenCL doesn't have an fprintf command.  How should we communicate this error when this is an OpenCL kernel?
   if (normalizeInputFlag && l_GSynNorm==0 && l_GSynExc != 0) {
      fprintf(stderr, "time = %f, k = %d, normalizeInputFlag is true but GSynNorm is zero and l_GSynExc = %f\n", timed, k, l_GSynExc);
      abort();
   };
   l_GSynExc /= (l_GSynNorm + (l_GSynNorm==0 ? 1 : 0));
   GSynExcEffective[k] = l_GSynExc;
   GSynInhEffective[k] = l_GSynInh;

   // add noise
   //
   excitatoryNoise[k] = 0.0f;
   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqE) {
      l_rnd = cl_random_get(l_rnd);
      excitatoryNoise[k] = params->noiseAmpE*cl_random_prob(l_rnd);
      l_GSynExc = l_GSynExc + excitatoryNoise[k];
   }

   inhibitoryNoise[k] = 0.0f;
   l_rnd = cl_random_get(l_rnd);
   float r = cl_random_prob(l_rnd);
   if (r < dt_sec*params->noiseFreqI) {
      l_rnd = cl_random_get(l_rnd);
      r = cl_random_prob(l_rnd);
      inhibitoryNoise[k] = params->noiseAmpI*r;
      l_GSynInh = l_GSynInh + inhibitoryNoise[k];
   }

   inhibNoiseB[k] = 0.0f;
   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqIB) {
      l_rnd = cl_random_get(l_rnd);
      inhibNoiseB[k] = params->noiseAmpIB*cl_random_prob(l_rnd);
      l_GSynInhB = l_GSynInhB + inhibNoiseB[k];
   }

   const float GMAX = FLT_MAX;

   // The portion of code below uses the newer method of calculating l_V.
   float G_E_initial, G_I_initial, G_IB_initial, G_E_final, G_I_final, G_IB_final;
   float tau_inf_initial, tau_inf_final, V_inf_initial, V_inf_final;

   G_E_initial = l_G_E + l_GSynExc;
   G_I_initial = l_G_I + l_GSynInh;
   G_IB_initial = l_G_IB + l_GSynInhB;
   tau_inf_initial = LCALIF_tauInf(tau, G_E_initial, G_I_initial, G_IB_initial, l_gapStrength);

   V_inf_initial = LCALIF_VmemInf(Vrest, Vexc, Vinh, VinhB, G_E_initial, G_I_initial, G_IB_initial, l_GSynGap, l_gapStrength);

   G_E_initial  = (G_E_initial  > GMAX) ? GMAX : G_E_initial;
   G_I_initial  = (G_I_initial  > GMAX) ? GMAX : G_I_initial;
   G_IB_initial = (G_IB_initial > GMAX) ? GMAX : G_IB_initial;

   float totalconductance = 1.0 + G_E_initial + G_I_initial + G_IB_initial + l_gapStrength;
   Vmeminf[k] = (Vrest + Vexc*G_E_initial + Vinh*G_I_initial + VinhB*G_IB_initial + l_GSynGap)/totalconductance;

   G_E_final = G_E_initial * decayE;
   G_I_final = G_I_initial * decayI;
   G_IB_final = G_IB_initial * decayIB;
   tau_inf_final = LCALIF_tauInf(tau, G_E_final, G_I_initial, G_IB_initial, l_gapStrength);
   V_inf_final = LCALIF_VmemInf(Vrest, Vexc, Vinh, VinhB, G_E_final, G_I_final, G_IB_final, l_GSynGap, l_gapStrength);

   float tau_slope = (tau_inf_final-tau_inf_initial)/dt;
   float f1 = tau_slope==0.0f ? EXP(-dt/tau_inf_initial) : powf(tau_inf_final/tau_inf_initial, -1/tau_slope);
   float f2 = tau_slope==-1.0f ? tau_inf_initial/dt*logf(tau_inf_final/tau_inf_initial+1.0f) :
                                 (1-tau_inf_initial/dt*(1-f1))/(1+tau_slope);
   float f3 = 1.0f - f1 - f2;
   l_V = f1*l_V + f2*V_inf_initial + f3*V_inf_final;
   
   l_G_E = G_E_final;
   l_G_I = G_I_final;
   l_G_IB = G_IB_final;

   //l_Vth updates according to traditional LIF rule in addition to the slow threshold adaptation
   //      See LCA_Equations.pdf in the documentation for a full description of the neuron adaptive firing threshold.
   
   Vadpt[k] = -60.0f;
   // Vadpt[k] += (dt/tauTHR) * ((integratedSpikeCount[k]/tauO) - targetRatekHz) * (Vscale/targetRatekHz);
   // float Vadpt_floor = params->Vrest + 5.0f;
   // Vadpt[k] = Vadpt[k] < Vadpt_floor ? Vadpt_floor : Vadpt[k];

   l_Vth = Vadpt[k] + decayVth * (l_Vth - Vadpt[k]);
   
   bool fired_flag = (l_V > l_Vth);

   l_activ = fired_flag ? 1.0f                 : 0.0f;
   Vattained[k] = l_V; // Save the value of V before it drops due to the spike
   l_V     = fired_flag ? Vrest                : l_V;
   l_Vth   = fired_flag ? l_Vth + deltaVth     : l_Vth;
   l_G_IB  = fired_flag ? l_G_IB + deltaGIB    : l_G_IB;


   //integratedSpikeCount is the trace activity of the neuron, with an exponential decay
   integratedSpikeCount[k] = decayO * (l_activ + integratedSpikeCount[k]);

   //
   // These actions must be done outside of kernel
   //    1. set activity to 0 in boundary (if needed)
   //    2. update active indices
   //

   // store local variables back to global memory
   //
   rnd[k] = l_rnd;

   activity[kex] = l_activ;

   V[k]   = l_V;
   Vth[k] = l_Vth;

   G_E[k]  = l_G_E;
   G_I[k]  = l_G_I;
   G_IB[k] = l_G_IB;

#ifndef PV_USE_OPENCL
   } // loop over k
#endif

}
