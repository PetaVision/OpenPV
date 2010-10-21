//#include <OpenCL/opencl.h>

#include <stdio.h>

//
// update the state of an LIF layer
//

//
//// 4. change if () guard so that it uses kx and ky

__kernel void update_state (
    __global float * V,
    __global float * G_E,
    __global float * G_I,
    __global float * G_IB,
    __global float * phi,
    __global float * activity,
    const uint nx,
    const uint ny,
    const uint nf,
    const uint nPad)
{
   float tauInf, VmemInf;
   float expExcite, expInhib, expInhibB;
   float dt, tau;
   float Vrest, Vexc, Vinh, VinhB;   
   
   //ADD DT, HARD CODE IN VARIABLES, FIX stuff, implement and check by hand, and check boundaries
   
   const uint kx = get_global_id(0);
   const uint ky = get_global_id(1);
   const uint k  = kx + nx*ky;
   
   float one = 1.1f;

   VmemInf = 20.0f;   
   //tauInf = 3.141f;

   //Hard Coded Variables
   tau = 20.0;
   Vrest = -70.0;
   Vexc = 0.0;
   Vinh = -75.0;
   VinhB = -90.0;

   dt = 1.0f;
   tau = 1.0f;
   Vrest = 2.0f;
   Vexc = 2.0f;
   Vinh = 2.0f;
   VinhB = 2.0f;

   expExcite = 1.0f;
   expInhib = 2.0f;
   expInhibB = 2.0f;

   if (kx < nx && ky < ny) {
      //printf ("%f %f %f\n", kx, ky, k);
      G_E[k]  = 1 + G_E[k] ;phiExc[k] + G_E[k]  * expExcite;
      G_I[k]  = phiInh[k]; //phiInh[k] + G_I[k]  * expInhib;
      G_IB[k] = phiInhB[k] + G_IB[k] * expInhibB;
      tauInf  = (dt / tau) * (1 + G_E[k] + G_I[k] + G_IB[k]);
      VmemInf = ( Vrest + G_E[k] * Vexc + G_I[k] * Vinh + G_IB[k] * VinhB )
                / (1.0 + G_E[k] + G_I[k] + G_IB[k]);
      V[k] = VmemInf + (V[k] - VmemInf) * exp(-tauInf);

      //printf("G_I[%d]=%f\n", k, G_I[k]);
      //printf("kx=%f\n", kx);
      //printf("nx=%f\n", nx);

//      V[k] = 8.0f;
   }
}
