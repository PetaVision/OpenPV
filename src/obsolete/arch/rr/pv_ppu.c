#include "pv.h"

#include <stdio.h>
#include <stdlib.h>

void update_V_and_f(int nc, float phi_c[], float Vc[], float fc[], float I[],
					float Hc[], float hc[]);


/**
 * Phase II
 *
 * stream the neurons through the SPUs, O(N^2) operations
 *
 * a forall?, chunks go to available threads
 *
 * @s
 * @hc
 */
int update_partial_state(PVState* s, int hc)
{
    int k;

    float* x = s->loc.x;
    float* y = s->loc.y;
    float* o = s->loc.o;

    // TODO take into account extended border
    float* phi = s->phi;
    float* f   = s->events[hc].event;

    for (k = 0; k < N; k += CHUNK_SIZE) {
      update_phi(CHUNK_SIZE, N, &phi[k], &x[k], &y[k], &o[k], x, y, o, f);
      //if (DEBUG) fprintf(stderr, "  update chunk %d k=%d %f\n", k/CHUNK_SIZE, k, phi[0]);
    }

    //    if (DEBUG) printf("[%d] update_partial_state: eventmask is %p, hc_id is %d\n",
    //		      s->comm_id, s->events[hc].event, hc);

    return 0;
}


/**
 * Phase V
 *
 * update V (with image and noise) and f
 *
 * @s
 */
int update_state(PVState* s)
{
   // TODO - take into account extended border
   update_V_and_f(N, s->phi, s->V, s->events[0].event, s->I, s->H, s->h);
   return 0;
}



/**
 *
 * variables:
 * @x - x position of a neuron (in pixels?)
 * @y - y position of a neuron
 * @th - orientation of line segment that a neuron responds to
 * @f - firing event mask
 * @argc
 * @argv
 */
int ppu_main(int argc, char* argv[])
{
  float fmax = 0.0;
  float phimax = -1000.;
  float Vmax = -1000.;
  float* f_accum;
  char* filename;

  PVState s;

  int i, loop;

  float* x  = s.loc.x;
  float* y  = s.loc.y;
  float* o  = s.loc.o;
  float  x0 = s.loc.x0;
  float  y0 = s.loc.y0;

  float* phi = s.phi;
  float* I   = s.I;
  float* f   = s.events[0].event;
  float* V   = s.V;
  float* H   = s.H;
  float* h   = s.h;

  double t_start = 0.0;
  double t_elapsed = 0.0;

  /* initialize environment state variables (no MPI) */

  s.comm_id     = 0;
  s.comm_size   = 1;
  s.n_rows      = 1;
  s.n_cols      = 0;
  s.n_neighbors = 0;

  init_state_ppu(&s);

  filename = (char*) malloc(64);
  f_accum = (float*) malloc(N*sizeof(float));

  for (i = 0; i < N; i++) {
    f_accum[i] = 0.0;
  }

  t_start = MPI_Wtime();
  for (loop = 1; loop < 50; loop++) {

    update_partial_state(&s, 0);


    /*
     * Phase V
     *
     * update phi (with image and noise), V, and f
     *
     */


    fmax = 0.0;
    phimax = -10000.0;
    Vmax = -10000.;
    for (i = 0; i < N; i++) {
      if (fmax < f[i]) fmax = f[i];
      if (phimax < phi[i]) phimax = phi[i];
      if (Vmax < V[i]) Vmax = V[i];
      f_accum[i] += f[i];
    }

    sprintf(filename, "f%d", loop);
    pv_output(filename, fmax/2., x0, y0, x, y, o, f);

    sprintf(filename, "phi%d", loop);
    pv_output(filename, phimax/2., x0, y0, x, y, o, phi);

    update_V_and_f(N, phi, V, f, I, H, h);

    sprintf(filename, "f_accum%d", loop);
    pv_output(filename, 0.8, x0, y0, x, y, o, f_accum);

    /* dump the output */
    post(0.8, x0, y0, x, y, o, V);

    Vmax = 0.0;
    for (i = 0; i < N; i++) {
      if (Vmax < V[i]) Vmax = V[i];
    }

    sprintf(filename, "V%d", loop);
    pv_output(filename, Vmax/2., x0, y0, x, y, o, V);

    //  fprintf(stderr, "  updated total %d %f\n", k-1, phi[0]);
    fprintf(stderr, "loop=%d:  fmax=%f, phimax=%f, Vmax=%f\n", loop, fmax, phimax, Vmax);

  } // end loop iterate

  t_elapsed = MPI_Wtime() - t_start;
  //  fprintf(stderr,  "GFlops/s = %f\n", ((2.0*N/1.e9)*N) / et);

  return 0;
}


/**
 * update the membrane potential and firing event mask
 *
 * @nc is the number of neurons to process in this chunk
 * @phi_c is the partial membrane potential
 * @Vc is the membrane potential
 * @fc is the firing event mask
 * @I is the input image
 * @Hc
 * @hc
 */
void update_V_and_f(int nc, float phi_c[], float Vc[], float fc[], float I[],
                    float Hc[], float hc[])
{
  int i;
  float r = 0.0;
  float Vth = V_TH_0;
  float Vth_inh = V_TH_0_INH;

  const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;

  for (i = 0; i < nc; i++) {

    /* inhibition layer, uses same zucker weights */
    // TODO - noise, image contribution?
	if ( rand()*INV_RAND_MAX < NOISE_FREQ_INH ) {
	    r = NOISE_AMP_INH * 2 * ( rand() * INV_RAND_MAX - 0.5 );
	}

	Hc[i] += DT_d_TAU_INH*(r + phi_c[i] - Hc[i]);
    hc[i]  = ((Hc[i] - Vth_inh) > 0.0) ? 1.0 : 0.0;
    Hc[i] -= hc[i]*Hc[i];	// reset cells that fired

    if ( rand()*INV_RAND_MAX < NOISE_FREQ ) {
      //r = NOISE_AMP * 2 * ( rand() * INV_RAND_MAX - 0.5 );
      if (I[i] > 0.0) r = I[i];
      else r = 0.0;
      //      printf("adding noise, r = %f, phi = %f, %d\n", r, phi_c[i], i);
    }

    phi_c[i] -= COCIRC_SCALE*fc[i];		// remove self excitation

    Vc[i] += DT_d_TAU*(/*I[i]*/ + r + phi_c[i] - Vc[i] - INHIBIT_AMP*hc[i]);
    fc[i]  = ((Vc[i] - Vth) > 0.0) ? 1.0 : 0.0;
    Vc[i] -= fc[i]*Vc[i];	// reset cells that fired

    phi_c[i] = 0.0;
  }

}
