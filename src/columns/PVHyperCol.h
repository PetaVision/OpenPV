#ifndef HYPERCOL_H_
#define HYPERCOL_H_

#include "pv.h"

typedef struct PVHyperCol_
  {
    int comm_id;
    int comm_size;
    int n_rows;
    int n_cols;
    int n_neighbors;
    int neighbors[NUM_NEIGHBORS]; /* mapping from neighbor index to neighbor rank */

    float       row;
    float       col;
    float       x0;
    float       y0;

    int n_layers;
    struct PVLayer* layer[MAX_LAYERS];

    eventmask* remote_events; /* event masks from neighbors */

    unsigned char* event_store; /* storage for local bit masks */

    double mpi_wait_time;
  } PVHyperCol;

/* "Methods" */

PVHyperCol* pv_new_hypercol(int comm_id, int comm_size, int nsteps);

int pv_hypercol_begin_update(PVHyperCol* hc, int neighbor_index, int time_step);
int pv_hypercol_finish_update(PVHyperCol* hc, int time_step);

int pv_hypercol_row(PVHyperCol* hc, int comm_id);
int pv_hypercol_col(PVHyperCol* hc, int comm_id);

eventmask* column_event(PVHyperCol* hc, int conn_id);

int pv_neighbor_init(PVHyperCol* hc);

int neighbor_index(PVHyperCol* hc, int comm_id, int index);
int pv_number_neighbors(PVHyperCol* hc, int comm_id);


int pv_north(PVHyperCol* hc, int comm_id);
int pv_south(PVHyperCol* hc, int comm_id);
int pv_east(PVHyperCol* hc, int comm_id);
int pv_west(PVHyperCol* hc, int comm_id);

int pv_northwest(PVHyperCol* hc, int comm_id);
int pv_northeast(PVHyperCol* hc, int comm_id);
int pv_southwest(PVHyperCol* hc, int comm_id);
int pv_southeast(PVHyperCol* hc, int comm_id);

/* error reporting */

void error_msg(PVHyperCol* s, int err, char* loc);

#endif /*HYPERCOL_H_*/
