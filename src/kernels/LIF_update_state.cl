#define EXP expf

//
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void LIF_update_state(
    const float time, 
    const float dt,
    const LIF_params * params,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL float * V,
    CL_MEM_GLOBAL float * Vth,
    CL_MEM_GLOBAL float * G_E,
    CL_MEM_GLOBAL float * G_I,
    CL_MEM_GLOBAL float * G_IB,
    CL_MEM_GLOBAL float * phiExc,
    CL_MEM_GLOBAL float * phiInh,
    CL_MEM_GLOBAL float * phiInhB,
    CL_MEM_GLOBAL float * activity)
{
#ifndef PV_USE_OPENCL
for (int k = 0; k < nx*ny*nf; k++) {
#else   
   int k = get_global_id(0);
#endif

   register int kex = kIndexExtended(k, nx, ny, nf, nb);

   //
   // kernel (nonheader part) begins here
   //

   // local param variables
   float tau, tauE, tauI, tauIB, Vrest, VthRest, Vexc, Vinh, VinhB, tauVth, deltaVth;

   const float GMAX = 10.0;

   // local variables
   float l_activ;

   float l_V   = V[k];
   float l_Vth = Vth[k];

   float l_G_E  = G_E[k];
   float l_G_I  = G_I[k];
   float l_G_IB = G_IB[k];

   float l_phiExc  = phiExc[k];
   float l_phiInh  = phiInh[k];
   float l_phiInhB = phiInhB[k];

   // temporary arrays
   float tauInf, VmemInf;

   //
   // start of LIF2_update_exact_linear
   //

   // define local param variables
   //
   tau   = params->tau;
   tauE  = params->tauE;
   tauI  = params->tauI;
   tauIB = params->tauIB;

   Vrest = params->Vrest;
   Vexc  = params->Vexc;
   Vinh  = params->Vinh;
   VinhB = params->VinhB;

   tauVth   = params->tauVth;
   VthRest  = params->VthRest;
   deltaVth = params->deltaVth;

//   call add_noise(l, dt);

   l_G_E  = l_phiExc  + l_G_E *EXP(-dt/tauE );
   l_G_I  = l_phiInh  + l_G_I *EXP(-dt/tauI );
   l_G_IB = l_phiInhB + l_G_IB*EXP(-dt/tauIB);

   tauInf  = (dt/tau) * (1.0 + l_G_E + l_G_I + l_G_IB);
   VmemInf = (Vrest + l_G_E*Vexc + l_G_I*Vinh + l_G_IB*VinhB)
           / (1.0 + l_G_E + l_G_I + l_G_IB);

   l_V = VmemInf + (l_V - VmemInf)*EXP(-tauInf);

   //
   // start of LIF2_update_finish
   //

   l_phiExc  = 0.0f;
   l_phiInh  = 0.0f;
   l_phiInhB = 0.0f;

   l_Vth = VthRest + (l_Vth - VthRest)*EXP(-dt/tauVth);

   //
   // start of update_f
   //

   l_G_E  = (l_G_E  > GMAX) ? GMAX : l_G_E;
   l_G_I  = (l_G_I  > GMAX) ? GMAX : l_G_I;
   l_G_IB = (l_G_IB > GMAX) ? GMAX : l_G_IB;

   l_activ = activity[kex];

   l_activ = (l_V > l_Vth) ? 1.0f           : 0.0f;
   l_V     = (l_V > l_Vth) ? Vrest          : l_V;
   l_Vth   = (l_V > l_Vth) ? l_Vth + deltaVth : l_Vth;
   l_G_IB  = (l_V > l_Vth) ? l_G_IB + 1.0f    : l_G_IB;

   //
   // These actions must be done outside of kernel
   //    1. set activity to 0 in boundary (if needed)
   //    2. update active indices
   //

   // store local variables back to global memory
   //
   activity[kex] = l_activ;

   V[k]   = l_V;
   Vth[k] = l_Vth;

   G_E[k]  = l_G_E;
   G_I[k]  = l_G_I;
   G_IB[k] = l_G_IB;

   phiExc[k]  = l_phiExc;
   phiInh[k]  = l_phiInh;
   phiInhB[k] = l_phiInhB;

#ifndef PV_USE_OPENCL
   } // loop over k
#endif

}
