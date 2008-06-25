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
    int i, j, k, t;

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
    V = l->phi;
    f = l->phi;

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
                V[k] = 0.0;
                f[k] = 0.0;

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
    float* buf = (float*) malloc( 6*N*sizeof(float));

    // TODO - assume event mask type is a float for now
    assert(sizeof(eventtype_t) == sizeof(float));

    l->x = buf + 0*N;
    l->y = buf + 1*N;
    l->o = buf + 2*N;

    l->phi = buf + 3*N;
    l->V = buf + 4*N;
    l->f = (eventtype_t*) (buf + 5*N);

    return 0;
  }

int pv_layer_send(PVLayer* l, int col_index)
  {
    return 0;
  }
