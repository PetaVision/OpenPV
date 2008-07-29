#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>
#include <layers/inhibit.h>
#include <layers/zucker.h>


#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef INHIBIT_ON
static float* buffer_get(int *index, eventtype_t *inhib_buffer[]);
static void buffer_put(int *index,eventtype_t *inhib_buffer[],float* phi, int n);

#define YPOS_FROM_IDX(idx) ( (idx)/(NX*NO))
#define XPOS_FROM_IDX(idx) ( ((idx)/NO) % NX)
#define O_FROM_IDX(idx) ( (idx) % NO)

int inhibit_update( PVLayer *l) 
{
  int i; 
  float *x,*y,*o;
  float *H, *phi_i, *phi_h, *phi_g; // phi_i input for inhibitory cell, phi_h output from inhibitory cell 
  eventtype_t *h, *hb;
  x = l->x;
  y = l->y;
  o = l->o;
  H = l->H;
  phi_i = l->phi_i;
  phi_h = l->phi_h;
  phi_g = l->phi_g;
  h=l->h;
  //f=l->f;

   
  //Update potential of inhib cells
  update_V(l->n_neurons, phi_i, phi_g, H, h, DT_d_TAU_INH, MIN_H, NOISE_FREQ_INH, NOISE_AMP_INH, SCALE_GAP);
  update_f(l->n_neurons,H, h, V_TH_0_INH);
 

  //Get delayed h
  hb = buffer_get(&(l->buffer_index_get), l->inhib_buffer);

  //now update inhibition contribution to excitory potential (phi_h)
  for(i=0;i<l->n_neurons; i+=CHUNK_SIZE)
    {
	update_phi( i, CHUNK_SIZE, l->n_neurons,  phi_h,
		  hb, INHIB_R2, SIG_I_D_x2, SIG_I_P_x2, SCALE_INH, INHIB_FRACTION_I, INHIBIT_SCALE_I);
    }
  
  //Put h in buffer
  buffer_put(&(l->buffer_index_put),l->inhib_buffer, h, l->n_neurons); 
 
  
  return 0;
}


static eventtype_t* buffer_get(int *index,eventtype_t *inhib_buffer[])
{
  eventtype_t* f = inhib_buffer[*index];
  (*index)++;
  if ((*index)==INHIB_DELAY)
    (*index)=0;
  return f;
}

static void buffer_put(int *index, eventtype_t *inhib_buffer[],eventtype_t *f, int n)
{
  int k;
  for(k=0; k<n; k++)
    { 
      inhib_buffer[*index][k] = f[k];
    }
  (*index)++;
  if ((*index)==INHIB_DELAY)
    (*index)=0;
  return;
}
#endif

void update_phi(int start_idx, int nc, int np, float phi_h[],
		eventtype_t hp[], int boundary, float sig_d2, float sig_p2, float scale,
		float inhib_fraction, float inhibit_scale)
{
  int i, j, ii, jj;

  // Each neuron is identified by location (xp/xc), iterated by i and j,
  // and orientation (thp/thc), iterated by ii and jj

  for (j = start_idx; j < (start_idx+np); j+=NO) {		// loop over all x,y locations

    for (jj = 0; jj < NO; jj++) {	// loop over all orientations

      if (hp[j+jj]==0.0) 		// If this neuron didn't fire, skip it.
	continue;
      

      for (i = 0; i < nc; i+=NO) 
	{	// loop over other neurons, first by x,y

	  float dx, dy, d2, gd, gt, ww;
	  int inner = 1;

	// use periodic (or mirror) boundary conditions	
	// Calc euclidean distance between neurons.

	dx = XPOS_FROM_IDX(j) - XPOS_FROM_IDX(i);
	dx = fabs(dx) > NX/2 ? -(dx/fabs(dx))*(NX - fabs(dx)) : dx; // PBCs
	dy = YPOS_FROM_IDX(j) - YPOS_FROM_IDX(i);
	dy = fabs(dy) > NY/2 ? -(dy/fabs(dy))*(NY - fabs(dy)) : dy;
	d2 = dx*dx + dy*dy;		// d2=sqr of euclidean distance	

/* 	printf("d2= %f ",d2); */
/* 	printf("R2= %f ", EXCITE_R2); */
	//check if neuron is within boundary- 2 options
	//1. assign 1 or 0 
	//2. check and kick out if outside boundary
	//inner =(d2 <= boundary) ? 1 : 0;
	if (d2> boundary) {
	  continue;
	}

	float gr = 1.0;
	float atanx2;
	float chi;
	
	// Calc angular diff btw this orientation and angle of adjoining line
	// 2nd term is theta(i,j) (5.1) from ParentZucker89
	atanx2 = O_FROM_IDX(j+jj)*DTH - RAD_TO_DEG_x2*atan2f(dy,dx);

	gd = expf(-d2/sig_d2);	// normalize dist for weight multiplier

	for (ii = 0; ii < NO; ii++) {	// now loop over each orienation

	  chi = atanx2 + O_FROM_IDX(i+ii)*DTH;	// Calc int. angle of this orienation 
	  chi = chi + 360.0f;		// range correct: (5.3) from ParentZucker89
	  chi = fmodf(chi,180.0f);
	  if (chi >= 90.0f) chi = 180.0f - chi;

	  gt = expf(-chi*chi/sig_p2); // normalize angle multiplier 

	  // Calculate and apply connection efficacy/weight 
	  ww = gd*(gt - inhib_fraction);
	  ww = (ww < 0.0) ? ww*inhibit_scale : ww;
	  phi_h[i+ii] = phi_h[i+ii] + scale*ww*inner;//*gr*hp[j+jj];
	} // i
      } // ii
    } // for jj
  } // for j
  

  return;
}




