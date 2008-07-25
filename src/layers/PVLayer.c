#include "pv.h"
#include "PVLayer.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

static int layer_alloc_mem(PVLayer* l);

/**
 * Constructor for a PVLayer
 */
PVLayer* pv_new_layer(PVHyperCol* hc, int index, int nx, int ny, int no)
  {

    float xc, yc, oc;
    int i, j, k, t, p;

    float *x, *y, *o;
    float *phi, *V;
    eventtype_t *f;
    


    float x0 = hc->x0;
    float y0 = hc->y0;

    PVLayer* l = (PVLayer*) malloc(sizeof(PVLayer));
  
    layer_alloc_mem(l);

    l->index = index;
    l->n_neurons = nx*ny*no;
    l->parent = hc;

    x = l->x;
    y = l->y;
    o = l->o;

    phi = l->phi;
    V = l->V;
    f = l->f;

#ifdef INHIBIT_ON
    float *phi_h, *phi_g, *phi_i, *H;
    eventtype_t *h;
    // float* inhib_buffer[INHIB_DELAY];
    l->buffer_index_get=0;
    l->buffer_index_put=0;
    phi_h = l->phi_h;
    phi_i = l->phi_i;
    phi_g = l->phi_g;
    H = l->H;
    h= l->h;

#endif 
   
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

                x[k] = xc - x0;
                y[k] = yc - y0;
                o[k] = oc;

                phi[k] = 0.0;
                float Vinit=(rand()/(float)RAND_MAX)*(V_TH_0-MIN_V)+MIN_V;
		V[k] = Vinit;
                
		f[k] = (float) (rand()/(float)RAND_MAX)<0.01;

#ifdef INHIBIT_ON		
		phi_h[k] = 0.0;
		phi_i[k] = 0.0;
		phi_g[k] = 0.0;
		h[k] = (float) (rand()/(float)RAND_MAX)<0.01;
		float Hinit= (rand()/(float) RAND_MAX)*(V_TH_0_INH-MIN_H)+MIN_H;
		H[k] = Hinit;
		for (p=0; p<INHIB_DELAY; p++)
		  l->inhib_buffer[p][k]= 0.0;
#endif
		k = k + 1;
	      } // t < no
          } // i < nx
      } // j < ny

    return l;
  }

/**
 * allocate and initialize Layer variable
 */

static int layer_alloc_mem(PVLayer* l)
  {
#ifdef INHIBIT_ON
    float* buf = (float*) malloc((11+INHIB_DELAY)*N*sizeof(float)); 
#else
    float* buf = (float*) malloc( 6*N*sizeof(float));
#endif
    // TODO - assume event mask type is a float for now
    assert(sizeof(eventtype_t) == sizeof(float));

    l->x = buf + 0*N;
    l->y = buf + 1*N;
    l->o = buf + 2*N;

    l->phi = buf + 3*N;
    l->V = buf + 4*N;
    l->f = (eventtype_t*) (buf + 5*N);

#ifdef INHIBIT_ON
    int i;
    l->H = buf + 6*N;
    l->phi_h = buf + 7*N;
    l->phi_i = buf + 8*N;
    l->phi_g = buf + 9*N;
    l->h = (eventtype_t*) (buf + 10*N);
    for(i=0; i<INHIB_DELAY; i++)
      l->inhib_buffer[i] = (eventtype_t*)buf +(10+ i)*N;
#endif
    return 0;
  }

int pv_layer_send(PVLayer* l, int col_index)
  {
    return 0;
  }

