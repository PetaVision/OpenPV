#include "pv.h"
#include "PVLayer.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static int layer_alloc_mem(PVLayer* l);

/**
 * Constructor for a PVLayer
 */
PVLayer* pv_new_layer(PVHyperCol* hc, int index, int nx, int ny, int no, int nk)
  {

    float xc, yc, oc, kappac;
    int i, j, k, t, p, q;

    float *x, *y, *o, *kappa; 
    float *phi, *V;
    eventtype_t *f;
    
    if(nk == 0) //set nk to min 1 before continuing
      nk = 1;

    float x0 = hc->x0;
    float y0 = hc->y0;
    
    PVLayer* l = (PVLayer*) malloc(sizeof(PVLayer));
    l->n_neurons = nx*ny*no*nk;
#ifdef INHIBIT_ON
    l->n_neuronsi= nx*ny*no;
#endif
    layer_alloc_mem(l);

    l->index = index;

    l->parent = hc;

    x = l->x;
    y = l->y;
    o = l->o;
 
    kappa = l->kappa;

    phi = l->phi;
    V = l->V;
    f = l->f;

#ifdef INHIBIT_ON
    float *xi, *yi, *oi;
    float *phi_h, *phi_g, *phi_i, *phiII, *H;
    eventtype_t *h;
    int ki=0;
    // float* inhib_buffer[INHIB_DELAY];
    xi = l->xi;
    yi = l->yi;
    oi = l->oi;
    l->buffer_index_get=0;
    l->buffer_index_put=0;
    phi_h = l->phi_h;
    phi_i = l->phi_i;
    phi_g = l->phi_g;
    phiII = l->phiII;
    H = l->H;
    h= l->h;

#endif 

    char filename[90];
    sprintf( filename, OUTPUT_PATH);
    strncat(filename, "/V_prob.txt",12);
    FILE* fo= fopen(filename,"w");
    if (fo== NULL)
      printf("error");
    k = 0;
    for (j = 0; j < ny; j++)
      {
        yc = y0 + j*DY;
        for (i = 0; i < nx; i++)
          {
            xc = x0 + i*DX;
            for (t = 0; t < no; t++)
              {
                oc = t*DTH;
#ifdef INHIBIT_ON 
		xi[ki] = xc - x0;
		yi[ki] = yc - y0;
		oi[ki] = oc;
		phi_i[ki] = 0.0; 
		phi_g[ki] = 0.0;
		h[ki] =(float) (rand()/(float)RAND_MAX)<0.01;
		float Hinit = (rand()/(float) RAND_MAX)*(V_TH_0_INH-MIN_H)+MIN_H;
		H[ki] =Hinit;
		phiII[ki] = 0.0;
		for (p=0; p<INHIB_DELAY; p++)
		  l->inhib_buffer[p][ki]= (float) (rand()/(float)RAND_MAX)<0.01;
#endif
		for( q = 0; q < nk; q++)
		  {
		    kappac = q*DK;

		    x[k] = xc - x0;
		    y[k] = yc - y0;
		    o[k] = oc;
		    kappa[k] = kappac;

		    phi[k] = 0.0;
		    float Vinit=(rand()/(float)RAND_MAX)*(V_TH_0-MIN_V)+MIN_V;
		    if(fo!=NULL)
		      fprintf( fo,"%f\n",(DT_d_TAU*Vinit/V_TH_0));

		    V[k] = Vinit;
		    
		    f[k] = (float) (rand()/(float)RAND_MAX)<0.01;
		    
#ifdef INHIBIT_ON		
		    phi_h[k] = 0.0;
#endif
		    k++;
		  }//q < nk
#ifdef INHIBIT_ON
		ki++;
#endif
	      } // t < no
	  } // i < nx
      } // j < ny
    
   /*  i=18; */
/*     j=18; */
/*     t=2; */
/*     q=2; */
/*     k= t+i*NO+j*NO*NX; */
/*     h[k] = 1.0; */
/*     k= q+k*NK; */
/*     f[k]=0.0;  */

    if(fo!=NULL)     
      fclose(fo); 
    return l;
  }

/**
 * allocate and initialize Layer variable
 */

static int layer_alloc_mem(PVLayer* l)
  {
#ifdef INHIBIT_ON
    float* buf = (float*) malloc(8*l->n_neurons*sizeof(float)+(8+INHIB_DELAY)*l->n_neuronsi*sizeof(float)); 
#else
    float* buf = (float*) malloc( 7*l->n_neurons*sizeof(float));
#endif
    // TODO - assume event mask type is a float for now
    assert(sizeof(eventtype_t) == sizeof(float));

    l->x = buf + 0*l->n_neurons;
    l->y = buf + 1*l->n_neurons;
    l->o = buf + 2*l->n_neurons;
    l->kappa = buf + 3*l->n_neurons;
    l->phi = buf + 4*l->n_neurons;
    l->V = buf + 5*l->n_neurons;
    l->f = (eventtype_t*) (buf + 6*l->n_neurons);

#ifdef INHIBIT_ON
    int i;
    l->phi_h = buf + 7*l->n_neurons; 
    l->xi = buf + 8*l->n_neurons + 0*l->n_neuronsi;
    l->yi = buf + 8*l->n_neurons + 1*l->n_neuronsi;
    l->oi = buf + 8*l->n_neurons + 2*l->n_neuronsi;
    l->H =  buf + 8*l->n_neurons + 3*l->n_neuronsi;
    l->phi_i = buf + 8*l->n_neurons + 4*l->n_neuronsi;
    
    l->phi_g = buf + 8*l->n_neurons + 5*l->n_neuronsi;
    l->phiII = buf + 8*l->n_neurons + 6*l->n_neuronsi; 
    l->h = (eventtype_t*) (buf + 8*l->n_neurons + 7*l->n_neuronsi);
    for(i=0; i<INHIB_DELAY; i++)
      l->inhib_buffer[i] = (eventtype_t*)(buf + 8*l->n_neurons + (8+i)*l->n_neuronsi);
#endif
    return 0;
  }

int pv_layer_send(PVLayer* l, int col_index)
  {
    return 0;
  }

