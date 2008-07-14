#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>

#include <math.h>
#include <stdlib.h>

static inline int update_V(int n, float phi[], float V[], float f[]);
static inline int update_f(int n, float V[], float f[]);

void update_phi(int nc, int np, float phi_c[], float xc[], float yc[],
     float thc[], float xp[], float yp[], float thp[], float fp[]);

/**
 * Constructor for a Zucker layer (PVLayer).
 * 
 * This layer just contains the firing events for an input image (currently a circle with clutter).
 * 
 */
PVLayer* pv_new_layer_zucker(PVHyperCol* hc, int index, int nx, int ny, int no)
  {
    PVLayer* l = pv_new_layer(hc, index, nx, ny, no);
    return l;
  }

int pv_layer_begin_update(PVLayer* l, int neighbor_index, int time_index)
  {
    int k;

    float* x = l->x;
    float* y = l->y;
    float* o = l->o;

    float* phi = l->phi;
    float* f = l->f;

    for (k = 0; k < N; k += CHUNK_SIZE)
      {
        update_phi(CHUNK_SIZE, N, &phi[k], &x[k], &y[k], &o[k], x, y, o, f);
        // if (DEBUG) fprintf(stderr, "  update chunk %d k=%d %f\n", k/CHUNK_SIZE, k, phi[0]);
      }

    // if (DEBUG) printf("[%d] update_partial_state: eventmask is %p, hc_id is %d\n",
    //                   s->comm_id, s->events[hc].event, hc);

    return 0;
  }

/**
 * Add feed forward contribution to partial membrane potential
 */
int pv_layer_add_feed_forward(PVLayer* l, PVLayer* llow, int neighbor_index, int time_index)
  {
    int i;
    
    // TODO - add lateral feeds (need list of connections)
    int n = l->n_neurons;
    float w = 1.0;
    
    float* phi = l->phi;
    float* fl  = llow->f;

    for (i = 0; i < n; i++)
      {
        phi[i] += w*fl[i];
      }
    
    return 0;
  }

/**
 * Finish updating a neuron layer
 */
int pv_layer_finish_update(PVLayer* l, int time_index)
  {
    update_V(N, l->phi, l->V, l->f);
    update_f(l->n_neurons, l->V, l->f);
    return 0;
  }

/**
 * update the partial sums for membrane potential
 *
 * Iterates over all neurons and updates postsynaptic membrane potentials,
 * conditionally (for performance) based on whether a given neuron fired.
 *
 * Uses chunking, so first triple (xc,yc,thc) identify current chunk subset
 * next triple (xp,yp,thp) represents all neurons 
 *
 * nc is the number of neurons to process in this chunk
 * np is the total number of neurons on this processor (size of event mask fp)
 */
void update_phi(int nc, int np, float phi_c[], float xc[], float yc[],
		float thc[], float xp[], float yp[], float thp[], float fp[])
{
  int i, j, ii, jj;
  

  // Each neuron is identified by location (xp/xc), iterated by i and j,
  // and orientation (thp/thc), iterated by ii and jj

  for (j = 0; j < np; j+=NO) {		// loop over all x,y locations

    for (jj = 0; jj < NO; jj++) {	// loop over all orientations

      if (fp[j+jj]==0.0) 		// If this neuron didn't fire, skip it.
	continue;

      for (i = 0; i < nc; i+=NO) {	// loop over other neurons, first by x,y
	float dx, dy, d2, gd, gt, ww;
	float gr = 1.0;
	float atanx2;
	float chi;

	// use periodic (or mirror) boundary conditions	
	// Calc euclidean distance between neurons.
	dx = xp[j] - xc[i];
	dx = fabs(dx) > NX/2 ? -(dx/fabs(dx))*(NX - fabs(dx)) : dx; // PBCs
	dy = yp[j] - yc[i];
	dy = fabs(dy) > NY/2 ? -(dy/fabs(dy))*(NY - fabs(dy)) : dy;
	
	// Calc angular diff btw this orientation and angle of adjoining line
	// 2nd term is theta(i,j) (5.1) from ParentZucker89
	atanx2 = thp[j+jj] - RAD_TO_DEG_x2*atan2f(dy,dx);

	d2 = dx*dx + dy*dy;		// d2=sqr of euclidean distance
	gd = expf(-d2/SIG_C_D_x2);	// normalize dist for weight multiplier

	for (ii = 0; ii < NO; ii++) {	// now loop over each orienation

	  chi = atanx2 + thc[i+ii];	// Calc int. angle of this orienation 
	  chi = chi + 360.0f;		// range correct: (5.3) from ParentZucker89
	  chi = fmodf(chi,180.0f);
	  if (chi >= 90.0f) chi = 180.0f - chi;

	  gt = expf(-chi*chi/SIG_C_P_x2); // normalize angle multiplier 

	  // Calculate and apply connection efficacy/weight 
	  ww = COCIRC_SCALE*gd*(gt - INHIB_FRACTION);
	  ww = (ww < 0.0) ? ww*INHIBIT_SCALE : ww;
	  phi_c[i+ii] = phi_c[i+ii] + ww;//*gr*fp[j+jj];
	} // i
      } // ii
    } // for jj
  } // for j
  
}

/**
 * update the membrane potential
 *
 * n is the number of neurons to process in this chunk
 * phi is the partial membrane potential
 * V is the membrane potential
 * f is the firing event mask
 * I is the input image
 */
static inline int update_V(int n, float phi[], float V[], float f[])
  {
    int i;
    float r = 0.0;
    // float Vth_inh= V_TH_0_INH;

    const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;

    for (i = 0; i < n; i++)
      {

        /* inhibition layer, uses same zucker weights */

        // TODO - noise, image contribution?
        // if (rand()*INV_RAND_MAX < NOISE_FREQ_INH)
        //   {
        //     r = NOISE_AMP_INH * 2 * ( rand() * INV_RAND_MAX - 0.5 );
        //   }

        // H[i] += DT_d_TAU_INH*(r + phi[i] - H[i]);
        // h[i] = ((H[i] - Vth_inh) > 0.0) ? 1.0 : 0.0;
        // H[i] -= h[i]*H[i]; // reset cells that fired

        if (rand()*INV_RAND_MAX < NOISE_FREQ)
          {
            r = NOISE_AMP * 2 * ( rand() * INV_RAND_MAX - 0.5 );
            //r = 0.0;
            // TODO - if noise only from image add to feed forward contribution
            // if (I[i] > 0.0) r = I[i];
            //      printf("adding noise, r = %f, phi = %f, %d\n", r, phi[i], i);
          }

        phi[i] -= COCIRC_SCALE*f[i]; // remove self excitation

        V[i] += DT_d_TAU*(r + phi[i] - V[i] /*- INHIBIT_AMP*h[i]*/);
        phi[i] = 0.0;
      }
    
    return 0;
  }

/**
 * update firing event mask
 *
 * n is the number of neurons to process in this chunk
 * V is the membrane potential
 * f is the firing event mask
 * I is the input image
 */
static inline int update_f(int n, float V[], float f[])
  {
    int i;
    float Vth= V_TH_0;

    for (i = 0; i < n; i++)
      {
        f[i] = ((V[i] - Vth) > 0.0) ? 1.0 : 0.0;
        V[i] -= f[i]*V[i]; // reset cells that fired
      }
    
    return 0;
  }
