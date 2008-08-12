#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>
#include <layers/inhibit.h>
#include <layers/zucker.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>



/* void update_phi(int nc, int np, float phi_c[], float xc[], float yc[], */
/*      float thc[], float xp[], float yp[], float thp[], float fp[]);  */

/**
 * Constructor for a Zucker layer (PVLayer).
 * 
 * This layer just contains the firing events for an input image (currently a circle with clutter).
 * 
 */
PVLayer* pv_new_layer_zucker(PVHyperCol* hc, int index, int nx, int ny, int no, int nk)
  {
    PVLayer* l = pv_new_layer(hc, index, nx, ny, no, nk);
    return l;
  }

int pv_layer_begin_update(PVLayer* l, int neighbor_index, int time_index)
  {
    int k;

    float* x = l->x;
    float* y = l->y;
    float* o = l->o;
#ifdef INHIBIT_ON
    float* xi = l->xi;
    float* yi = l->yi;
    float* oi = l->oi;   
#endif
    float* kappa = l->kappa;

    float* phi = l->phi;
    eventtype_t* f = l->f;

    for (k = 0; k < l->n_neurons; k += CHUNK_SIZE)
      {
        update_phi(CHUNK_SIZE, l->n_neurons, NO, NO, NK, NK, &phi[k], &x[k], &y[k], &o[k], &kappa[k], x, y, o, f, EXCITE_R2, SIG_C_D_x2, SIG_C_P_x2, COCIRC_SCALE, INHIB_FRACTION, INHIBIT_SCALE, 1,SIG_C_K_x2, 1);

        // if (DEBUG) fprintf(stderr, "  update chunk %d k=%d %f\n", k/CHUNK_SIZE, k, phi[0]);
      }
  
#ifdef INHIBIT_ON

    eventtype_t *h = l->h;
    float *phi_g = l->phi_g;
   
    inhibit_update(l, time_index);
#endif

    // if (DEBUG) printf("[%d] update_partial_state: eventmask is %p, hc_id is %d\n",
    //                   s->comm_id, s->events[hc].event, hc);

    return 0;
  }

/**
 * Add feed forward contribution to partial membrane potential
 */
int pv_layer_add_feed_forward(PVLayer* l, PVLayer* llow, int neighbor_index, int time_index)
  {
    int i, j;
    
    // TODO - add lateral feeds (need list of connections)
    int n = llow->n_neurons;
    float w = 1.0;
    float wi = 0.0; 
    float* phi_i = l->phi_i;
    float* phi = l->phi;
    float* fl  = llow->f;

#if POISSON_INPUT
    long prob;
    prob = (float)(POISSON_RATE_TIMESTEPS*exp(-1.0 * POISSON_RATE_TIMESTEPS))*RAND_MAX;

#endif //POISSON_INPUT

    for (i = 0; i < n; i+=NK) {
#if POISSON_INPUT
		// Make input fire probabilistically based on Poisson distr
		// It's cheating to put it here, but we don't currently
		// do dynamic updating for the input layer.
		float fired = rand() > prob ? 0.0 : 1.0;
		for(j=0; j<NK; j++)
		      phi[i+j] +=  fired*w*fl[i+j];

		phi_i[(i/NK)]+= fired*wi*fl[i];
#else
		// Non-poisson: edge detectors fire each timestep
		for(j=0; j<NK; j++)
		  phi[i] += w*fl[i];
		
		phi_i[(i/NK)]+= wi*fl[i];
#endif //POISSON_INPUT
		
	

      }
    
    return 0;
  }

/**
 * Finish updating a neuron layer
 */
int pv_layer_finish_update(PVLayer* l, int time_index)
  {
#ifdef INHIBIT_ON
    char filename1[64];
    char filename2[64];

   
    //finish inhibit update
    update_V(l->n_neuronsi, l->phi_i, l->phi_g, l->H, l->h, DT_d_TAU_INH, MIN_H, NOISE_FREQ_INH, NOISE_AMP_INH, SCALE_GAP);
    update_f(l->n_neuronsi,l->H, l->h, V_TH_0_INH); 
   
    //finish excite update
    update_V(N, l->phi,l->phi_h, l->V, l->f, DT_d_TAU, MIN_V, NOISE_FREQ, NOISE_AMP, COCIRC_SCALE);
    update_f(l->n_neurons, l->V, l->f, V_TH_0); 
   

    
#else 
    float phi_h[N]={0.0}; 
    update_V(l->n_neurons, l->phi, phi_h, l->V, l->f, DT_d_TAU, MIN_V, NOISE_FREQ, NOISE_AMP, COCIRC_SCALE);
    update_f(l->n_neurons, l->V, l->f, V_TH_0);  
#endif
 
  
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
/* void update_phi(int nc, int np, float phi_c[], float xc[], float yc[], */
/* 		float thc[], float xp[], float yp[], float thp[], float fp[]) */
/* { */
/*   int i, j, ii, jj; */
  

/*   // Each neuron is identified by location (xp/xc), iterated by i and j, */
/*   // and orientation (thp/thc), iterated by ii and jj */

/*   for (j = 0; j < np; j+=NO) {		// loop over all x,y locations */

/*     for (jj = 0; jj < NO; jj++) {	// loop over all orientations */

/*       if (fp[j+jj]==0.0) 		// If this neuron didn't fire, skip it. */
/* 	continue; */

/*       for (i = 0; i < nc; i+=NO) {	// loop over other neurons, first by x,y */
/* 	float dx, dy, d2, gd, gt, ww; */
/* 	float gr = 1.0; */
/* 	float atanx2; */
/* 	float chi; */

/* 	// use periodic (or mirror) boundary conditions	 */
/* 	// Calc euclidean distance between neurons. */
/* 	dx = xp[j] - xc[i]; */
/* 	dx = fabs(dx) > NX/2 ? -(dx/fabs(dx))*(NX - fabs(dx)) : dx; // PBCs */
/* 	dy = yp[j] - yc[i]; */
/* 	dy = fabs(dy) > NY/2 ? -(dy/fabs(dy))*(NY - fabs(dy)) : dy; */
	
/* 	// Calc angular diff btw this orientation and angle of adjoining line */
/* 	// 2nd term is theta(i,j) (5.1) from ParentZucker89 */
/* 	atanx2 = thp[j+jj] - RAD_TO_DEG_x2*atan2f(dy,dx); */

/* 	d2 = dx*dx + dy*dy;		// d2=sqr of euclidean distance */
/* 	gd = expf(-d2/SIG_C_D_x2);	// normalize dist for weight multiplier */

/* 	for (ii = 0; ii < NO; ii++) {	// now loop over each orienation */

/* 	  chi = atanx2 + thc[i+ii];	// Calc int. angle of this orienation  */
/* 	  chi = chi + 360.0f;		// range correct: (5.3) from ParentZucker89 */
/* 	  chi = fmodf(chi,180.0f); */
/* 	  if (chi >= 90.0f) chi = 180.0f - chi; */

/* 	  gt = expf(-chi*chi/SIG_C_P_x2); // normalize angle multiplier  */

/* 	  // Calculate and apply connection efficacy/weight  */
/* 	  ww = COCIRC_SCALE*gd*(gt - INHIB_FRACTION); */
/* 	  ww = (ww < 0.0) ? ww*INHIBIT_SCALE : ww; */
/* 	  phi_c[i+ii] = phi_c[i+ii] + ww;//*gr*fp[j+jj]; */
/* 	} // i */
/*       } // ii */
/*     } // for jj */
/*   } // for j */
  
/* } */

/**
 * update the membrane potential
 *
 * n is the number of neurons to process in this chunk
 * phi is the partial membrane potential
 * V is the membrane potential
 * f is the firing event mask
 * I is the input image
 */
inline int update_V(int n, float phi[],float phi_h[], float V[], float f[], float dt_d_tau, float v_min, float noise_freq, float noise_amp, float scale)
  {
    int i;
    float r = 0.0;
    // float Vth_inh= V_TH_0_INH;

    const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;

    for (i = 0; i < n; i++)
      {
	if (rand()*INV_RAND_MAX < noise_freq)
          {
	    r = noise_amp * 2 * ( rand() * INV_RAND_MAX - 0.5 );
            //r = 0.0;
            // TODO - if noise only from image add to feed forward contribution
            // if (I[i] > 0.0) r = I[i];
            //      printf("adding noise, r = %f, phi = %f, %d\n", r, phi[i], i);
          }
	else
	  r = 0.0;

	// phi[i] -= scale*f[i]; // remove self excitation
#ifdef INHIBIT_ON
	V[i] += dt_d_tau*(r + phi[i] - V[i] +phi_h[i]);

	phi_h[i]=0.0;
	phi[i]=0.0;
	V[i]= (V[i]<v_min)? v_min : V[i];
#else
        V[i] += dt_d_tau*(r + phi[i] - V[i] /*- INHIBIT_AMP*h[i]*/);
	phi[i] = 0.0;
#endif
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
inline int update_f(int n, float V[], float f[], float Vth)
  {
    int i;

    for (i = 0; i < n; i++)
      {
        f[i] = ((V[i] - Vth) > 0.0) ? 1.0 : 0.0;
        V[i] -= f[i]*V[i]; // reset cells that fired
      }
    
    return 0;
  }
void debug_filer (char* path,eventtype_t *h, float *phi1, float *phi2, int time, int n)
{
  char* filename;
  int i, j;
  int u=0;
  int path_len = strlen(OUTPUT_PATH) + 1 + strlen(path);
  int path_size = (path_len+8)*sizeof(char);	// contains a little extra space

  filename = malloc(path_size);
  if ( filename == NULL ) {
    printf("ERROR:pv_output: malloc(%d bytes) failed\n", path_size);
    exit(1);
  }
  strcpy(filename, OUTPUT_PATH);
  strncat(filename, "/", 2);
  strcat(filename, path);
  strncat(filename, ".txt", 5);
  FILE* fid;
  if(time == 0)
    { fid= fopen(filename, "w");}
  else
    { fid= fopen(filename, "a"); }
  if (fid != NULL)
    {
      fprintf(fid, "******************************************\n");
      fprintf(fid, "#%d TIME LOOP\n", time);
      fprintf(fid, "******************************************\n\n");
      fprintf(fid, " FIRING MASK\n\n");
      for(i=0; i<n; i++)
	{
	 
	  if(h[i]==0)
	    continue;
	  if(u == 10)
	    {
	      fprintf(fid, "%d:%f\n",i, h[i]);
	      u=0;
	      continue;
	    }
	  fprintf(fid, "%d:%f    ",i,h[i]);	 
	  u++;
	   	 
	} 
      fprintf(fid, "\n\n PHI 1 MASK \n\n");
      for(i=0; i<n; i++)
	{
	  if(phi1[i] == 0.0)
	    continue;
	  if(u == 10)
	    {
	      fprintf(fid, "%d:%f\n",i, phi1[i]);
	      u=0;
	      continue;
	    }
	  fprintf(fid, "%d:%f    ",i,phi1[i]);	 
	  u++;	  	 
	}
      fprintf(fid, "\n\n PHI 2 MASK \n\n");
      for(i=0; i<n; i++)
	{
	  if(phi2[i]==0)
	    continue;
	  if(u == 10)
	    {
	      fprintf(fid, "%d:%f\n",i, phi2[i]);
	      u=0;
	      continue;
	    }
	  fprintf(fid, "%d:%f    ",i,phi2[i]);	 
	  u++;	  	 
	}  
    }
  return;
}
