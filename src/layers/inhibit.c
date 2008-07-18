#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>
#include<layers/inhibit.h>


#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef INHIBIT_ON
static float* buffer_get(int *index, float *inhib_buffer[]);
static void buffer_put(int *index,float *inhib_buffer[],float* phi, int n);




int inhibit_update( PVLayer *l) 
{
  const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;
  int k,i,j,kk,ii; 
  float r;
  float *x,*y,*o;
  float *H, *phi, *phi_hi, *phi_ho; // phi_hi input for inhibitory cell, phi_ho output from inhibitory cell 
  eventtype_t *h;
  x = l->x;
  y = l->y;
  o = l->o;
  H = l->H;
  phi = l->phi;
  phi_ho = l->phi_h;
  h=l->h;
  
  // printf("before dump\n");
/*   dump = (float*) malloc(6*N*sizeof(float)); */
/*   phi_hi = dump + 1*N; */
/*   if(dump == NULL || phi_hi == NULL) */
/*     printf("Error in memory allocation of pointer in inhibit.c."); */
/*   printf("assignment works\n"); */

  //Get delayed phi from buffer
  phi_hi = buffer_get(&(l->buffer_index_get), l->inhib_buffer);
  //free up memory when not in use
  // free(dump);
  
  //update phi input

 /*  for(i=0; i<l->n_neurons; i+=CHUNK_SIZE) */
/*     { */
/*       update_phi( CHUNK_SIZE, l->n_neurons,  &phi_hi[i], &x[i], &y[i], */
/* 		    &o[i], x, y, o, h, GAP_R2, SIG_G_D_x2, SIG_G_P_x2, SCALE_GAP); */
/*     } */


  for(j=0; j<l->n_neurons; j++) 
   {
     //TODO - noise, image contribution?
     if (rand()*INV_RAND_MAX < NOISE_FREQ_INH)
       {
	 r = NOISE_AMP_INH * 2 * ( rand() * INV_RAND_MAX - 0.5 );
       }
     
     /* inhibition layer, uses same zucker weights as exicitory, but delayed + they get gap junction input */
     H[j] += DT_d_TAU_INH*(r + phi_hi[j] - H[j]);
     h[j] = ((H[j] - V_TH_0_INH) > 0.0) ? 1.0 : 0.0;
     H[j] -= h[j]*H[j]; // reset cells that fired
     phi[j]=(E_TO_I_SCALE/COCIRC_SCALE)*phi[j];    
   }    


 //now update inhibition contribution to excitory potential (phi_h) 
 for(kk=0;kk<l->n_neurons; kk+=CHUNK_SIZE)
   {

     update_phi( CHUNK_SIZE, l->n_neurons,  &phi_ho[kk], &x[kk], &y[kk],
		   &o[kk], x, y, o, h, INHIB_R2, SIG_I_D_x2, SIG_I_P_x2, SCALE_INH);
   }

 buffer_put(&(l->buffer_index_put),l->inhib_buffer, phi, l->n_neurons); 

 for(k=0;k<l->n_neurons;k++)
   phi[k]=0.0;
 return 0;
}


static float* buffer_get(int *index,float *inhib_buffer[])
{
  float* h = inhib_buffer[*index];
  *index++;
  if (*index==10)
    *index=0;
  return h;
}

static void buffer_put(int *index, float *inhib_buffer[],float *phi, int n)
{
  int k;
  for(k=0; k<n; k++)
    { 
      inhib_buffer[*index][k] = phi[k];
    }
  *index++;
  if (*index==10)
    *index=0;
  return;
}
#endif
void update_phi(int nc, int np, float phi_h[], float xc[], float yc[],
		  float thc[], float xp[], float yp[], float thp[], float hp[], int boundary, float sig_d2, float sig_p2, float scale)
{
  int i, j, ii, jj;
  

  // Each neuron is identified by location (xp/xc), iterated by i and j,
  // and orientation (thp/thc), iterated by ii and jj

  for (j = 0; j < np; j+=NO) {		// loop over all x,y locations

    for (jj = 0; jj < NO; jj++) {	// loop over all orientations

      if (hp[j+jj]==0.0) 		// If this neuron didn't fire, skip it.
	continue;
      

      for (i = 0; i < nc; i+=NO) 
	{	// loop over other neurons, first by x,y

	  float dx, dy, d2, gd, gt, ww;
	  int inner = 1;

	// use periodic (or mirror) boundary conditions	
	// Calc euclidean distance between neurons.

	dx = xp[j] - xc[i];
	dx = fabs(dx) > NX/2 ? -(dx/fabs(dx))*(NX - fabs(dx)) : dx; // PBCs
	dy = yp[j] - yc[i];
	dy = fabs(dy) > NY/2 ? -(dy/fabs(dy))*(NY - fabs(dy)) : dy;
	d2 = dx*dx + dy*dy;		// d2=sqr of euclidean distance	

/* 	printf("d2= %f ",d2); */
/* 	printf("R2= %f ", EXCITE_R2); */
	//check if neuron is within boundary- 2 options
	//1. assign 1 or 0 
	//2. check and kick out if outside boundary
	//inner =(d2 <= boundary) ? 1 : 0;
	if (d2> boundary)
	  {

	  continue;
	 
	  }
	float gr = 1.0;
	float atanx2;
	float chi;


	
	// Calc angular diff btw this orientation and angle of adjoining line
	// 2nd term is theta(i,j) (5.1) from ParentZucker89
	atanx2 = thp[j+jj] - RAD_TO_DEG_x2*atan2f(dy,dx);


	gd = expf(-d2/sig_d2);	// normalize dist for weight multiplier

	for (ii = 0; ii < NO; ii++) {	// now loop over each orienation

	  chi = atanx2 + thc[i+ii];	// Calc int. angle of this orienation 
	  chi = chi + 360.0f;		// range correct: (5.3) from ParentZucker89
	  chi = fmodf(chi,180.0f);
	  if (chi >= 90.0f) chi = 180.0f - chi;

	  gt = expf(-chi*chi/sig_p2); // normalize angle multiplier 

	  // Calculate and apply connection efficacy/weight 
	  ww = scale*gd*(gt - INHIB_FRACTION);
	  ww = (ww < 0.0) ? ww*INHIBIT_SCALE : ww;
	  phi_h[i+ii] = phi_h[i+ii] + ww*inner;//*gr*hp[j+jj];
	} // i
      } // ii
    } // for jj
  } // for j
  

  return;
}



//#endif
