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


int inhibit_update( PVLayer *l, int time_index) 

{
  int i; 
  float *x,*y,*o, *kappa, *xi,*yi,*oi;
  float *H, *phi_i, *phi_h, *phi_g, *phiII; // phi_i input for inhibitory cell, phi_h output from inhibitory cell 
  eventtype_t *f, *h, *hb;
  x = l->x;
  y = l->y;
  o = l->o;
  xi = l->xi;
  yi = l->yi;
  oi = l->oi;
  kappa = l->kappa;

  H = l->H;
  phi_i = l->phi_i;
  phi_h = l->phi_h;
  phi_g = l->phi_g;
  phiII = l->phiII;
  h=l->h;
  f=l->f;
  

  
  //update phi input from excitatory cells to inhibitory(phi_i)
  for(i=0; i<l->n_neuronsi; i+=(CHUNK_SIZE/NK))
    {
      update_phi( (CHUNK_SIZE/NK), l->n_neurons, NO, NO, 1, NK, &phi_i[i], &xi[i], &yi[i],
		  &oi[i],kappa, x, y, o, f, E2I_R2, SIG_E2I_D_x2, SIG_E2I_P_x2, E_TO_I_SCALE, INHIB_FRACTION_E2I, INHIBIT_SCALE_E2I, 1, SIG_C_K_x2, 0);
    }  
  
  
  //update phi input from gap junctions(phi_g)
  for(i=0; (i<l->n_neurons/NK); i+=(CHUNK_SIZE/NK))
    {
      update_phi( (CHUNK_SIZE/NK), l->n_neuronsi,NO, NO, 1, 1, &phi_g[i], &xi[i], &yi[i],
		  &oi[i], kappa, xi, yi, oi, h,GAP_R2, SIG_G_D_x2, SIG_G_P_x2, SCALE_GAP, INHIB_FRACTION_G, INHIBIT_SCALE_G, 0, SIG_C_K_x2, 1);
    }
  
 

  //Get delayed h
  hb = buffer_get(&(l->buffer_index_get), l->inhib_buffer);

  //now update inhibition contribution to excitory potential (phi_h)
  for(i=0;i<l->n_neurons; i+=CHUNK_SIZE)
    {
      
      update_phi( CHUNK_SIZE, l->n_neuronsi, NO, NO, NK, 1, &phi_h[i], &x[i], &y[i],
		  &o[i], kappa, xi, yi, oi, hb, INHIB_R2, SIG_I_D_x2, SIG_I_P_x2, SCALE_INH, INHIB_FRACTION_I, INHIBIT_SCALE_I, 0, SIG_C_K_x2, 0);

    }
 
 for(i=0;i<l->n_neuronsi; i+=(CHUNK_SIZE/NK))
    {
      
      update_phi( (CHUNK_SIZE/NK), l->n_neuronsi, NO, NO, 1, 1, &phiII[i], &xi[i], &yi[i],
		  &oi[i], kappa, xi, yi, oi, hb, INHIBI_R2, SIG_II_D_x2, SIG_II_P_x2, SCALE_INHI, INHIB_FRACTION_II, INHIBIT_SCALE_II, 0, SIG_C_K_x2, 1);
    }
    

  char filename1[64];
  sprintf(filename1, "inhibiting1b");
  debug_filer(filename1, l->h, l->phiII, l->phi_g, time_index, l->n_neuronsi);

  //Put h in buffer
  buffer_put(&(l->buffer_index_put),l->inhib_buffer, h, l->n_neuronsi); 
  
  //Update partial potential of inhib cells(combine phiII and phi_g)
  for (i=0; i<l->n_neuronsi; i++)
    {
      phi_g[i]= phi_g[i] + phiII[i];
      phiII[i]=0.0;
    }  
  
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


/*upate phi function- variable meaning*/
/*phi_h is the phi that needs updating*/
/*nc and c quantities are for the cluster being proccessed*/
/*np and p quantities are for all the neurons on the proccessor*/
/*boundary is the boundary constant for this connection*/
/*sig_d2 and sig_p2 are the sigmas for this connection*/
/*scale is the scale of the weight for this connection*/
/*inhib_fraction is the precintile of largest weighted connection*/
/*inhibit_scale scales down the other connection*/
/*curve is 1 for a connection with curvature and 0 for those with out*/
/*self is 1 for groups which include the neuron and 0 are for exclusive groups*/

void update_phi(int nc, int np, int noc, int nop, int nkc, int nkp, float phi_h[], float xc[], float yc[],
		float thc[],float kappap[], float xp[], float yp[], float thp[], float hp[], int boundary, float sig_d2, float sig_p2, float scale, 
		float inhib_fraction, float inhibit_scale, int curve, float sig_k2, int self)

{
  int i, j, ii, jj, jjj, iii;
  int curve2= (curve== 0)? 1 : 0;  
  int self2 = (self == 0)? 1 : 0;
  // Each neuron is identified by location (xp/xc), iterated by i and j,
  // and orientation (thp/thc), iterated by ii and jj


  if(fabs(scale) < MIN_DENOM)return; 
  for (j = 0; j < np; j+=(nop*nkp)) {		// loop over all x,y locations
    for (jj = 0; jj < (nop*nkp); jj+= nkp)
      {	// loop over all orientations
	for (jjj = 0; jjj< nkp; jjj++)
	  {
	    if (hp[j+jj+jjj]==0.0) 		// If this neuron didn't fire, skip it.
	      continue;
	    
	    
	    for (i = 0; i < nc; i+=(nkc*noc)) 
	      {	// loop over other neurons, first by x,y
		
		
		float dx, dy, d2, gd, gt, ww;
		int inner = 1;
		float selfcorrect;
		
		// use periodic (or mirror) boundary conditions	
		// Calc euclidean distance between neurons.
		
		dx = -xp[j] + xc[i];
		dx = fabs(dx) > NX/2 ? -(dx/fabs(dx))*(NX - fabs(dx)) : dx; // PBCs
		dy = -yp[j] + yc[i];
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
		float gr;
		float atanx2;
		float chi;
		
		
		
		/*** restrict d2 to band around radius of cocircle through points i,j ***/ 
		float k_cocirc; 
		float radp= (DEG_TO_RAD)*thp[j+jj];
		float dxP = ( dx * cos(radp) + dy * sin(radp) );
		float dyP = ( dy * cos(radp) - dx * sin(radp) );
		float z = ( 2 * dyP );
		// float sgnz = z / ( MIN_DENOM + fabs(z) );
		k_cocirc =fabs(z) / ( MIN_DENOM + d2 ); // fix denominator == 0
		gr = fabs(exp( -curve*pow((k_cocirc - fabs(kappap[j+jj+jjj])), 2 ) /sig_k2));//(curve2 + curve);//average connections
		    
		// Calc angular diff btw this orientation and angle of adjoining line
		// 2nd term is theta(i,j) (5.1) from ParentZucker89
		if( dx==0 && dy==0)
		  atanx2 = thp[j+jj];
		else
		  atanx2 = thp[j+jj] + RAD_TO_DEG_x2*atan2f(dyP,dxP);
		
		
		gd = expf(-d2/sig_d2);	// normalize dist for weight multiplier
		
		for (ii = 0; ii < (noc*nkc); ii+=nkc) 
		  {	// now loop over each orienation
		    
		    chi = atanx2 - thc[i+ii];	// Calc int. angle of this orienation 
		    chi = chi + 360.0f;		// range correct: (5.3) from ParentZucker89
		    chi = fmodf(chi,180.0f);
		    if (chi >= 90.0f) chi = 180.0f - chi;
		    
		    gt = expf(-chi*chi/sig_p2); // normalize angle multiplier 
		    
		    
		    
		    // gd=1.0;
		    //gr=1.0;
		    // Calculate and apply connection efficacy/weight 
		    ww = fabs(gd*gr)*(fabs(gt) - inhib_fraction);
		    ww = (ww < 0.0) ? ww*inhibit_scale : ww;
		    selfcorrect = (self*d2/(d2+MIN_DENOM) + self2);
		    for(iii = 0; iii<nkc; iii++)
		      { 
			phi_h[i+ii+iii] += scale*ww*inner*selfcorrect;//*hp[j+jj];
		      }//iii
		  } // ii
	      } // i
	  }// for jjj
      } // for jj
  } // for j
  
  return;
}



