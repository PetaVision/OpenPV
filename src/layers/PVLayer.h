#ifndef PV_LAYER_H_
#define PV_LAYER_H_

#include <columns/PVHyperCol.h>

/**
 * A PVLayer is a collection of neurons of a specific class 
 */
typedef struct PVLayer_
  {
    int index;          /* index of layer */
    int n_neurons;      /* number of neurons in layer */

    PVHyperCol* parent;
    
    struct PVLayer* layer_up;  /* pointer to layer below */
    struct PVLayer* layer_dn;  /* pointer to layer above */

    /* location and orientation */
    float* x;
    float* y;
    float* o;
    
    float* phi; /* potential for partial updates */
    float* V;   /* membrane potential */

    eventtype_t* f;       /* event mask */

  } PVLayer;

/* "Methods" */

PVLayer* pv_new_layer(PVHyperCol* hc, int index, int nx, int ny, int no);

int pv_layer_send(PVLayer* l, int col_index);

int pv_layer_begin_update(PVLayer* l, int neighbor_index, int time_index);

int pv_layer_add_feed_forward(PVLayer* l, PVLayer* llow, int neighbor_index, int time_index);

int pv_layer_finish_update(PVLayer* l, int time_index);

#endif /*PV_LAYER_H_*/
