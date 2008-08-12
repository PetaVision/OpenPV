#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>
#include <layers/retina.h>
#include <layers/zucker.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hc_init_layers(PVHyperCol* hc);

PVHyperCol* pv_new_hypercol(int comm_id, int comm_size, int nsteps, char *input_filename)
  {
    int my_seed;
    int nrows, ncols;
    float r;
    
    PVHyperCol* hc = (PVHyperCol*) malloc(sizeof(PVHyperCol));

    /* ensure that parameters make sense */

    assert(CHUNK_SIZE <= N);

    my_seed = (SEED + comm_id ) % RAND_MAX;
    srand(my_seed);
    
    r = sqrt(comm_size);
    nrows = (int) r;
    ncols = (int) comm_size/nrows;

    hc->comm_id = comm_id;
    hc->comm_size = nrows*ncols;

    hc->n_rows = nrows;
    hc->n_cols = ncols;

    hc->row = pv_hypercol_row(hc, hc->comm_id);
    hc->col = pv_hypercol_col(hc, hc->comm_id);
    hc->x0 = 0.0 + hc->col*DX*NX;
    hc->y0 = 0.0 + hc->row*DY*NY;

    hc->mpi_wait_time = 0.0;

    strcpy(hc->input_filename, input_filename);

    pv_neighbor_init(hc);
    
    hc_init_layers(hc);

    /* allocate memory for remote event masks */

    hc->remote_events = (eventmask*) malloc( (hc->n_neighbors)*sizeof(eventmask));
    if (hc->remote_events == NULL)
      {
        fprintf(stderr, "ERROR:init_state: malloc of hc->remote_events failed\n");
        exit(1);
      }

    // TODO - take into account multiple neighbors
    hc->event_store = (unsigned char*) malloc(nsteps * N/8);
    if (hc->event_store == NULL)
      {
        fprintf(stderr, "ERROR:init_state: malloc hc->event_store failed\n");
        exit(1);
      }

    if (DEBUG) printf("[%d] HyperCol: comm_size=%d, nrows=%d, ncols=%d\n",
                      hc->comm_id, hc->comm_size, hc->n_rows, hc->n_cols);

    if (DEBUG)
      printf("[%d] init_state: local eventmask is %p, n_neighbors is %d\n",
          hc->comm_id, hc->remote_events, hc->n_neighbors);

    return hc;
  }


static int hc_init_layers(PVHyperCol* hc)
  {
    // initially just retina and Zucker
    hc->n_layers = 2;

    // TODO - fix the circular dependencies in PVHyperCol and PVLayer
    hc->layer[0] = (struct PVLayer *) pv_new_layer_retina(hc, 0, NX, NY, NO, NK);
    hc->layer[1] = (struct PVLayer *) pv_new_layer_zucker(hc, 1, NX, NY, NO, NK);
    
    return 0;
  }

int pv_hypercol_row(PVHyperCol* hc, int comm_id)
  {
    return (int) comm_id/hc->n_cols;
  }

int pv_hypercol_col(PVHyperCol* hc, int comm_id)
  {
    return (comm_id - hc->n_cols * pv_hypercol_row(hc, comm_id) );
  }

int pv_hypercol_begin_update(PVHyperCol* hc, int ihc, int t)
  {
    int l;
    
    for (l = 0; l < hc->n_layers; l++)
      {
        // TODO - fix the circular dependencies in PVHyperCol and PVLayer
        pv_layer_send((PVLayer*) hc->layer[l], ihc);
      }

    // TODO - make layers a C++ object so that layer updates are polymorphic
    //for (l = 0; l < hc->n_layers; l++)
      //{
    //pv_layer_begin_update(hc->layer[l], ihc, t);
      //}
    // for now only update layer 1 (zucker)
    // TODO - fix the circular dependencies in PVHyperCol and PVLayer
    pv_layer_begin_update((PVLayer*) hc->layer[1], ihc, t);
    
    
    for (l = 1; l < hc->n_layers; l++)
      {
        // TODO - fix the circular dependencies in PVHyperCol and PVLayer
        pv_layer_add_feed_forward((PVLayer*) hc->layer[l], (PVLayer*) hc->layer[l-1], ihc, t);
      }

    return 0;
  }

int pv_hypercol_finish_update(PVHyperCol* hc, int t)
  {
    
    // TODO - make layers a C++ object so that layer updates are polymorphic
    // for now only update layer 1 (zucker)
    //for (l = 0; l < hc->n_layers; l++)
      //{
          //pv_layer_finish_update(hc->layer[l], t);
      //}
    // TODO - fix the circular dependencies in PVHyperCol and PVLayer
    pv_layer_finish_update((PVLayer*) hc->layer[1], t);
 
    return 0;
  }

eventmask* column_event(PVHyperCol* hc, int conn_id)
  {
    return &(hc->remote_events[conn_id+1]);
  }

void error_msg(PVHyperCol* hc, int err, char* loc)
  {
    if (err)
      {
        fprintf(stderr, "[%d] %s: ERROR(=%d) occurred in ???\n", hc->comm_id, loc, err);
      }
  }
