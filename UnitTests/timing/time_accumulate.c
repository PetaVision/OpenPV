#include "../../src/include/pv_common.h"
#include "../../src/include/pv_types.h"
#include "../../src/layers/PVLayer.h"

#include "clock.h"
#include <stdio.h>

// controls use of restrict C99 keyword
#define RESTRICT

#define NB 8
#define NF 4
#define NX 64
#define NY 64
#define NXP 4
#define NYP 4

#ifdef COMPILE_ME
static inline
void pvpatch_accumulate_old(PVPatch* phi, float a, PVPatch* weight)
{
   int k;
   float x, y, f;
   const float nx = phi->nx;
   const float ny = phi->ny;
   const float nf = phi->nf;
   const float sy = phi->sy;
   const float sf = phi->sf;

   // assume unit stride for w (densely packed)
   float* w = weight->data;

   for (y = 0; y < ny; y++) {
	//         float* v = phi->data + (int)(y*sy) + (int) (f*sf);
      float* v = phi->data + (int)nf*(NX+NB);
      for (k = 0; k < nf*nx; k += 4) {
         // there will be at least 4
         *v++ += a * (*w++);
         *v++ += a * (*w++);
         *v++ += a * (*w++);
         *v++ += a * (*w++);
      }
   }
}

static inline
void accumulate_c(float* RESTRICT v, float a, float* RESTRICT w)
{
   int k, x, y, f;

   const int nx = 4;
   const int ny = 4;
   const int nf = NF;
   const int sx = NF;
   const int sy = NF*(NX+NB);

   for (y = 0; y < ny; y++) {
	//         float* v = phi->data + (int)(y*sy) + (int) (f*sf);
      for (k = 0; k < nf*nx; k++) {
	 v[k] += a * w[k];
      }
   }
}

void mask_multiply(int nk, float* RESTRICT mw, float* RESTRICT m, float* RESTRICT w)
{
   int k;
   for (k = 0; k < nk; k++) {
      mw[k] = m[k] * w[k];
   }
}
#endif

#ifdef COMPRESS_PHI
static inline
void pvpatch_accumulate_loc(int nk, float* RESTRICT v, float a,
                            float* RESTRICT w, float* RESTRICT m)
{
   const float scale = 33.3;
   const float inv_scale = 1.0/scale;
   const float shift = 2.0;
   int k;

   for (k = 0; k < nk; k++) {
            v[k] = (((shift + scale*v[k]) + a*w[k]*m[k])
                  - shift) * inv_scale;
      // without mask
      //      v[k] = (((shift + scale*v[k]) + a*w[k])
      //                  - shift) * inv_scale;
   }
}
#else
static inline
void pvpatch_accumulate_loc(int nk, float* RESTRICT v, float a,
                            float* RESTRICT w, float* RESTRICT m)
{
   int k;
   for (k = 0; k < nk; k++) {
      v[k] = v[k] + a*w[k]*m[k];
   }
}
#endif


int main(int argc, char* argv[])
{
   int i, j, t, yw;
   double time, flops;

   int const nloops = 1000;

   float a = 1;
   float phi[NF*(NX+NB)*(NY+NB)] __attribute__ ((aligned));

   float w[NF*NXP*NYP]; // aligned attribute apparently not needed (and bad if wrong)
   float mw[NF*NXP*NYP];
   float mask[NF*NXP*NXP];

   PVPatch patch_phi, patch_w;

   const int nxp = 4;
   const int nyp = 4;
   const int nf  = NF;
   const int sf  = 1;
   const int sx  = NF;
   const int sy  = sx*nxp;

   //                flops *   patch_size  * grid_size
   //                  6,2 with no mask
#ifdef COMPRESS_PHI
   const int nflops =  7   * ((nf*nxp*nyp) * NX * NY);
#else
   const int nflops =  3   * ((nf*nxp*nyp) * NX * NY);
#endif

   pvpatch_init(&patch_w, nxp, nyp, nf, sx, sy, sf, w);

   pvpatch_init(&patch_phi, nxp, nyp, nf, sx, sx*(NX+NB), sf, phi);

   for (i = 0; i < nxp*nyp*nf; i++) {
      w[i] = 0.5;
      mask[i] = 2.0;
      mw[i] = mask[i]*w[i];
   }

   printf("starting loop....\n");

   start_clock();

   // only one feature fires so no loop over features in pre layer
   // however there is a loop over features in post layer

   for (t = 0; t < nloops; t++) {
      for (j = 0; j < NY; j++) {
         for (i = 0; i < NX; i++) {
           for (yw = 0; yw < nyp/4; yw++) {
              //accumulate_c(phi, a, w);
              //patch_phi.data = &phi[j][i][0];
              //hyperpatch_accumulate(&patch_phi, a, &patch_w);

              // on PPU these will be contiguous so can do nf*nxp*nyp
              // don't understand why can offset by variable amounts in units of 4

              // true patch offset to phi is i*sx + j*sy

              // this didn't work out
              // mask_multiply(nf*nxp*nyp, mw, mask, w);

              // WARNING - assumes nyp divisible by 4
              pvpatch_accumulate_loc(nf*nxp, phi + 0*sx + sy*(0+0),   a, w, mask+0);
              pvpatch_accumulate_loc(nf*nxp, phi + 3*sx + sy*(2+1),   a, w, mask+16);
              pvpatch_accumulate_loc(nf*nxp, phi + 16*sx + sy*(9+2),  a, w, mask+32);
	      pvpatch_accumulate_loc(nf*nxp, phi + 63*sx + sy*(63+3), a, w, mask+48);
	   }
	 }
      }
   }

   stop_clock();

   time = elapsed_time();
   flops = 1e-9 * nloops * nflops / time;
   printf("elapsed time is %f GFLOPS=%f\n", (float) time, (float) flops);
   printf("\njunk value is %f\n\n", phi[31]);

   return 0;
}


