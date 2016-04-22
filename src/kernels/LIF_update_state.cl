#include "LIF_params.h"
#include "cl_random.hcl"
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
float LIF_Vmem_derivative(
      const float Vmem,
      const float G_E,
      const float G_I,
      const float G_IB,
      const float V_E,
      const float V_I,
      const float V_IB,
      const float Vrest,
      const float tau) {
   float totalconductance = 1.0 + G_E + G_I + G_IB;
   float Vmeminf = (Vrest + V_E*G_E + V_I*G_I + V_IB*G_IB)/totalconductance;
   return totalconductance*(Vmeminf-Vmem)/tau;
}
//
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
// LIF_update_state_original uses an Euler scheme for V where the conductances over the entire timestep are taken to be the values calculated at the end of the timestep
// LIF_update_state_beginning uses a Heun scheme for V, using values of the conductances at both the beginning and end of the timestep.  Spikes in the input are applied at the beginning of the timestep.
// LIF_update_state_arma uses an auto-regressive moving average filter for V, applying the GSyn at the start of the timestep and assuming that tau_inf and V_inf vary linearly over the timestep.  See van Hateren, Journal of Vision (2005), p. 331.
//
CL_KERNEL
void LIF_update_state_original(
    const int nbatch,
    const int numNeurons,
    const float time, 
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    CL_MEM_CONST LIF_params * params,
    CL_MEM_GLOBAL taus_uint4 * rnd,
    CL_MEM_GLOBAL float * V,
    CL_MEM_GLOBAL float * Vth,
    CL_MEM_GLOBAL float * G_E,
    CL_MEM_GLOBAL float * G_I,
    CL_MEM_GLOBAL float * G_IB,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity)
{
   int k;

   const float exp_tauE    = EXP(-dt/params->tauE);
   const float exp_tauI    = EXP(-dt/params->tauI);
   const float exp_tauIB   = EXP(-dt/params->tauIB);
   const float exp_tauVth  = EXP(-dt/params->tauVth);

   const float dt_sec = .001 * dt;   // convert to seconds

#ifndef PV_USE_OPENCL

for (k = 0; k < nx*ny*nf*nbatch; k++) {
#else
   k = get_global_id(0);
#endif

   int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

   //
   // kernel (nonheader part) begins here
   //

   // local param variables
   float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

   const float GMAX = 10.0;

   // local variables
   float l_activ;

   taus_uint4 l_rnd = rnd[k];

   float l_V   = V[k];
   float l_Vth = Vth[k];

   float l_G_E  = G_E[k];
   float l_G_I  = G_I[k];
   float l_G_IB = G_IB[k];

   CL_MEM_GLOBAL float * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynInhB = &GSynHead[CHANNEL_INHB*nbatch*numNeurons];
   float l_GSynExc  = GSynExc[k];
   float l_GSynInh  = GSynInh[k];
   float l_GSynInhB = GSynInhB[k];
   
   // temporary arrays
   float tauInf, VmemInf;

   //
   // start of LIF2_update_exact_linear
   //

   // define local param variables
   //
   tau   = params->tau;
   Vexc  = params->Vexc;
   Vinh  = params->Vinh;
   VinhB = params->VinhB;
   Vrest = params->Vrest;

   VthRest  = params->VthRest;
   deltaVth = params->deltaVth;
   deltaGIB = params->deltaGIB;

   // add noise
   //

   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqE) {
      l_rnd = cl_random_get(l_rnd);
      l_GSynExc = l_GSynExc + params->noiseAmpE*cl_random_prob(l_rnd);
   }

   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqI) {
      l_rnd = cl_random_get(l_rnd);
      l_GSynInh = l_GSynInh + params->noiseAmpI*cl_random_prob(l_rnd);
   }

   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqIB) {
      l_rnd = cl_random_get(l_rnd);
      l_GSynInhB = l_GSynInhB + params->noiseAmpIB*cl_random_prob(l_rnd);
   }

   l_G_E  = l_GSynExc  + l_G_E *exp_tauE;
   l_G_I  = l_GSynInh  + l_G_I *exp_tauI;
   l_G_IB = l_GSynInhB + l_G_IB*exp_tauIB;
   
   l_G_E  = (l_G_E  > GMAX) ? GMAX : l_G_E;
   l_G_I  = (l_G_I  > GMAX) ? GMAX : l_G_I;
   l_G_IB = (l_G_IB > GMAX) ? GMAX : l_G_IB;

   tauInf  = (dt/tau) * (1.0 + l_G_E + l_G_I + l_G_IB);
   VmemInf = (Vrest + l_G_E*Vexc + l_G_I*Vinh + l_G_IB*VinhB)
           / (1.0 + l_G_E + l_G_I + l_G_IB);

   l_V = VmemInf + (l_V - VmemInf)*EXP(-tauInf);

   //
   // start of LIF2_update_finish
   //

   l_Vth = VthRest + (l_Vth - VthRest)*exp_tauVth;

   //
   // start of update_f
   //

   bool fired_flag = (l_V > l_Vth);

   l_activ = fired_flag ? 1.0f                 : 0.0f;
   l_V     = fired_flag ? Vrest                : l_V;
   l_Vth   = fired_flag ? l_Vth + deltaVth     : l_Vth;
   l_G_IB  = fired_flag ? l_G_IB + deltaGIB    : l_G_IB;

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

   GSynExc[k]  = 0.0f;
   GSynInh[k]  = 0.0f;
   GSynInhB[k] = 0.0f;

#ifndef PV_USE_OPENCL
   } // loop over k
#endif

}

CL_KERNEL
void LIF_update_state_beginning(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    CL_MEM_CONST LIF_params * params,
    CL_MEM_GLOBAL taus_uint4 * rnd,
    CL_MEM_GLOBAL float * V,
    CL_MEM_GLOBAL float * Vth,
    CL_MEM_GLOBAL float * G_E,
    CL_MEM_GLOBAL float * G_I,
    CL_MEM_GLOBAL float * G_IB,
    CL_MEM_GLOBAL float * GSynHead,
//    CL_MEM_GLOBAL float * GSynExc,
//    CL_MEM_GLOBAL float * GSynInh,
//    CL_MEM_GLOBAL float * GSynInhB,
    CL_MEM_GLOBAL float * activity)
{
   int k;

   const float exp_tauE    = EXP(-dt/params->tauE);
   const float exp_tauI    = EXP(-dt/params->tauI);
   const float exp_tauIB   = EXP(-dt/params->tauIB);
   const float exp_tauVth  = EXP(-dt/params->tauVth);

   const float dt_sec = .001 * dt;   // convert to seconds

#ifndef PV_USE_OPENCL

for (k = 0; k < nx*ny*nf*nbatch; k++) {
#else
   k = get_global_id(0);
#endif

   int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

   //
   // kernel (nonheader part) begins here
   //

   // local param variables
   float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

   const float GMAX = 10.0;

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

   CL_MEM_GLOBAL float * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
   CL_MEM_GLOBAL float * GSynInhB = &GSynHead[CHANNEL_INHB*nbatch*numNeurons];
   float l_GSynExc  = GSynExc[k];
   float l_GSynInh  = GSynInh[k];
   float l_GSynInhB = GSynInhB[k];

   //
   // start of LIF2_update_exact_linear
   //

   // define local param variables
   //
   tau   = params->tau;
   Vexc  = params->Vexc;
   Vinh  = params->Vinh;
   VinhB = params->VinhB;
   Vrest = params->Vrest;

   VthRest  = params->VthRest;
   deltaVth = params->deltaVth;
   deltaGIB = params->deltaGIB;

   // add noise
   //

   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqE) {
      l_rnd = cl_random_get(l_rnd);
      l_GSynExc = l_GSynExc + params->noiseAmpE*cl_random_prob(l_rnd);
   }

   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqI) {
      l_rnd = cl_random_get(l_rnd);
      l_GSynInh = l_GSynInh + params->noiseAmpI*cl_random_prob(l_rnd);
   }

   l_rnd = cl_random_get(l_rnd);
   if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqIB) {
      l_rnd = cl_random_get(l_rnd);
      l_GSynInhB = l_GSynInhB + params->noiseAmpIB*cl_random_prob(l_rnd);
   }

   // The portion of code below uses the newer method of calculating l_V.
   float G_E_initial, G_I_initial, G_IB_initial, G_E_final, G_I_final, G_IB_final;
   float dV1, dV2, dV;

   G_E_initial = l_G_E + l_GSynExc;
   G_I_initial = l_G_I + l_GSynInh;
   G_IB_initial = l_G_IB + l_GSynInhB;

   G_E_initial  = (G_E_initial  > GMAX) ? GMAX : G_E_initial;
   G_I_initial  = (G_I_initial  > GMAX) ? GMAX : G_I_initial;
   G_IB_initial = (G_IB_initial > GMAX) ? GMAX : G_IB_initial;

   G_E_final = G_E_initial*exp_tauE;
   G_I_final = G_I_initial*exp_tauI;
   G_IB_final = G_IB_initial*exp_tauIB;

   dV1 = LIF_Vmem_derivative(l_V, G_E_initial, G_I_initial, G_IB_initial, Vexc, Vinh, VinhB, Vrest, tau);
   dV2 = LIF_Vmem_derivative(l_V+dt*dV1, G_E_final, G_I_final, G_IB_final, Vexc, Vinh, VinhB, Vrest, tau);
   dV = (dV1+dV2)*0.5;
   l_V = l_V + dt*dV;

   l_G_E = G_E_final;
   l_G_I = G_I_final;
   l_G_IB = G_IB_final;

   l_Vth = VthRest + (l_Vth - VthRest)*exp_tauVth;
   // End of code unique to newer method.

   //
   // start of update_f
   //

   bool fired_flag = (l_V > l_Vth);

   l_activ = fired_flag ? 1.0f                 : 0.0f;
   l_V     = fired_flag ? Vrest                : l_V;
   l_Vth   = fired_flag ? l_Vth + deltaVth     : l_Vth;
   l_G_IB  = fired_flag ? l_G_IB + deltaGIB    : l_G_IB;

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

CL_KERNEL
void LIF_update_state_arma(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    CL_MEM_CONST LIF_params * params,
    CL_MEM_GLOBAL taus_uint4 * rnd,
    CL_MEM_GLOBAL float * V,
    CL_MEM_GLOBAL float * Vth,
    CL_MEM_GLOBAL float * G_E,
    CL_MEM_GLOBAL float * G_I,
    CL_MEM_GLOBAL float * G_IB,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity)
{
   int k;

   const float exp_tauE    = EXP(-dt/params->tauE);
   const float exp_tauI    = EXP(-dt/params->tauI);
   const float exp_tauIB   = EXP(-dt/params->tauIB);
   const float exp_tauVth  = EXP(-dt/params->tauVth);

   const float dt_sec = .001 * dt;   // convert to seconds

#ifndef PV_USE_OPENCL

   for (k = 0; k < nx*ny*nf*nbatch; k++) {
#else
   k = get_global_id(0);
   { // compound statement so indentation is consistent with the for loop when not using PV_USE_OPENCL
#endif

      int kex = kIndexExtendedBatch(k, nbatch, nx, ny, nf, lt, rt, dn, up);

      //
      // kernel (nonheader part) begins here
      //

      // local param variables
      float tau, Vrest, VthRest, Vexc, Vinh, VinhB, deltaVth, deltaGIB;

      const float GMAX = 10.0;

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

      CL_MEM_GLOBAL float * GSynExc = &GSynHead[CHANNEL_EXC*nbatch*numNeurons];
      CL_MEM_GLOBAL float * GSynInh = &GSynHead[CHANNEL_INH*nbatch*numNeurons];
      CL_MEM_GLOBAL float * GSynInhB = &GSynHead[CHANNEL_INHB*nbatch*numNeurons];
      float l_GSynExc  = GSynExc[k];
      float l_GSynInh  = GSynInh[k];
      float l_GSynInhB = GSynInhB[k];

      //
      // start of LIF2_update_exact_linear
      //

      // define local param variables
      //
      tau   = params->tau;
      Vexc  = params->Vexc;
      Vinh  = params->Vinh;
      VinhB = params->VinhB;
      Vrest = params->Vrest;

      VthRest  = params->VthRest;
      deltaVth = params->deltaVth;
      deltaGIB = params->deltaGIB;

      // add noise
      //

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqE) {
         l_rnd = cl_random_get(l_rnd);
         l_GSynExc = l_GSynExc + params->noiseAmpE*cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqI) {
         l_rnd = cl_random_get(l_rnd);
         l_GSynInh = l_GSynInh + params->noiseAmpI*cl_random_prob(l_rnd);
      }

      l_rnd = cl_random_get(l_rnd);
      if (cl_random_prob(l_rnd) < dt_sec*params->noiseFreqIB) {
         l_rnd = cl_random_get(l_rnd);
         l_GSynInhB = l_GSynInhB + params->noiseAmpIB*cl_random_prob(l_rnd);
      }

      // The portion of code below uses the newer method of calculating l_V.
      float G_E_initial, G_I_initial, G_IB_initial, G_E_final, G_I_final, G_IB_final;
      float tau_inf_initial, tau_inf_final, V_inf_initial, V_inf_final;

      G_E_initial = l_G_E + l_GSynExc;
      G_I_initial = l_G_I + l_GSynInh;
      G_IB_initial = l_G_IB + l_GSynInhB;
      tau_inf_initial = tau/(1+G_E_initial+G_I_initial+G_IB_initial);
      V_inf_initial = (Vrest+Vexc*G_E_initial+Vinh*G_I_initial+VinhB*G_IB_initial)/(1+G_E_initial+G_I_initial+G_IB_initial);

      G_E_initial  = (G_E_initial  > GMAX) ? GMAX : G_E_initial;
      G_I_initial  = (G_I_initial  > GMAX) ? GMAX : G_I_initial;
      G_IB_initial = (G_IB_initial > GMAX) ? GMAX : G_IB_initial;

      G_E_final = G_E_initial*exp_tauE;
      G_I_final = G_I_initial*exp_tauI;
      G_IB_final = G_IB_initial*exp_tauIB;

      tau_inf_final = tau/(1+G_E_final+G_I_final+G_IB_initial);
      V_inf_final = (Vrest+Vexc*G_E_final+Vinh*G_I_final+VinhB*G_IB_final)/(1+G_E_final+G_I_final+G_IB_final);

      float tau_slope = (tau_inf_final-tau_inf_initial)/dt;
      float f1 = tau_slope==0.0f ? EXP(-dt/tau_inf_initial) : powf(tau_inf_final/tau_inf_initial, -1/tau_slope);
      float f2 = tau_slope==-1.0f ? tau_inf_initial/dt*logf(tau_inf_final/tau_inf_initial+1.0f) :
                                    (1-tau_inf_initial/dt*(1-f1))/(1+tau_slope);
      float f3 = 1.0f - f1 - f2;
      l_V = f1*l_V + f2*V_inf_initial + f3*V_inf_final;

      l_G_E = G_E_final;
      l_G_I = G_I_final;
      l_G_IB = G_IB_final;

      l_Vth = VthRest + (l_Vth - VthRest)*exp_tauVth;
      // End of code unique to newer method.

      //
      // start of update_f
      //

      bool fired_flag = (l_V > l_Vth);

      l_activ = fired_flag ? 1.0f                 : 0.0f;
      l_V     = fired_flag ? Vrest                : l_V;
      l_Vth   = fired_flag ? l_Vth + deltaVth     : l_Vth;
      l_G_IB  = fired_flag ? l_G_IB + deltaGIB    : l_G_IB;

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
   }

}
