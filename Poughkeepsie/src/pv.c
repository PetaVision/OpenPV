#include "pv.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PARTIAL_OUTPUT 1

/* declaration of internal functions (should these be static) */

int output_partial_state(PVState* s, int time_step);
int output_final_state(PVState* s, int time_step);
void pv_output_events_circle(int step, float f[], float h[]);
void pv_output_on_circle(int step, const char* name, float max, float buf[]);

static void print_result( int length, int cycles, double time )
{
  double bandwidth, clock_prec;

  if (time < 0.000001) return;

  clock_prec = MPI_Wtick();
  bandwidth = (length * clock_prec * cycles) / (1024.0 * 1024.0) / (time * clock_prec);
  printf( "%8d\t%.6f\t%.4f MB/s\n", length, time / cycles, bandwidth );
}


int main(int argc, char* argv[])
{
    int c, t;
    int n_time_steps = 1;
    int hc_id, comm_id, comm_size;
    double tstart, tend;
    PVState s;

    MPI_Request req[NUM_NEIGHBORS];

    comm_init(&argc, &argv, &comm_id, &comm_size);

    if (argc == 2) {
      n_time_steps = atoi(argv[1]);
    }

    init_state(&s, comm_id, comm_size, n_time_steps);

    if (DEBUG) printf("[%d] PetaVision: comm_size=%d, nrows=%d, ncols=%d\n",
		      s.comm_id, s.comm_size, s.n_rows, s.n_cols);

    /* time loop */

    tstart = MPI_Wtime();

    for (t = 0; t < n_time_steps; t++) {
        send_state(&s, req);

	/* update with local events first */
        update_partial_state(&s, 0);

        /* loop over neighboring columns */
	for (c = 0; c < s.n_neighbors; c++) {
	    recv_state(&s, req, &hc_id);		/* hc_id is the index of neighbor */
            update_partial_state(&s, hc_id + 1);	/* 0 is local index, 1 first neighbor */
        }

#ifdef PARTIAL_OUTPUT
	output_partial_state(&s, t);
#endif

        /* complete update (new V and local event mask) */
	update_state(&s);

#ifdef PARTIAL_OUTPUT
	output_state(&s, t);
#endif
    }

    tend = MPI_Wtime();
    if (s.comm_id == 0) {
        printf("[0] ");
        print_result(NUM_MASK_EVENTS, n_time_steps, tend - tstart );
	printf("[0] mpi_wait_time = %lf\n", s.mpi_wait_time);
    }

    output_final_state(&s, n_time_steps);
    comm_finalize();

    return 0;
}


static int hasWesternNeighbor(PVState* s, int comm_id)
{
    return comm_id % s->n_cols;
}


static int hasEasternNeighbor(PVState* s, int comm_id)
{
    return (comm_id + 1) % s->n_cols;
}


static int hasNorthernNeighbor(PVState* s, int comm_id)
{
    return ((comm_id + s->n_cols) > (s->comm_size - 1)) ? 0 : 1;
}


static int hasSouthernNeighbor(PVState* s, int comm_id)
{
    return ((comm_id - s->n_cols) < 0) ? 0 : 1;
}


static int number_neighbors(PVState* s, int comm_id)
{
    int n = 0;

    int hasWest  = hasWesternNeighbor(s, comm_id);
    int hasEast  = hasEasternNeighbor(s, comm_id);
    int hasNorth = hasNorthernNeighbor(s, comm_id);
    int hasSouth = hasSouthernNeighbor(s, comm_id);
  
    if (hasNorth > 0) n += 1;
    if (hasSouth > 0) n += 1;

    if (hasWest > 0) {
        n += 1;
	if (hasNorth > 0) n += 1;
	if (hasSouth > 0) n += 1;
    }

    if (hasEast > 0) {
        n += 1;
	if (hasNorth > 0) n += 1;
	if (hasSouth > 0) n += 1;
    }

    return n;
}


static void init_neighbors(PVState* s)
{
    int i, n;
    int n_neighbors = 0;

    /* initialize neighbors lists */

    s->n_neighbors = number_neighbors(s, s->comm_id);

    for (i = 0; i < NUM_NEIGHBORS; i++) {
        s->neighbors[i] = 0;
        n = neighbor_index(s, s->comm_id, i);
	if (n >= 0) {
	  s->neighbors[n_neighbors++] = n;
          if (DEBUG) printf("[%d] init_neighbors: neighbor[%d] of %d is %d, i = %d\n",
			    s->comm_id, n_neighbors-1, s->n_neighbors, n, i);
	}
    }
    assert(s->n_neighbors == n_neighbors);
}


int init_state(PVState* s, int comm_id, int comm_size, int nsteps)
{
    int nrows, ncols;
    char filename[64];
    float r;

    /* ensure that parameters make sense */

    assert(CHUNK_SIZE <= N);

    int my_seed = ( SEED + comm_id ) % RAND_MAX;
    srand(my_seed);

    r = sqrt(comm_size);
    nrows = (int) r;
    ncols = (int) comm_size/nrows;

    s->comm_id = comm_id;
    s->comm_size = nrows*ncols;

    s->n_rows = nrows;
    s->n_cols = ncols;

    s->loc.row = pv_row(s, s->comm_id);
    s->loc.col = pv_col(s, s->comm_id);
    s->loc.x0  = 0.0 + s->loc.col*DX*NX;
    s->loc.y0  = 0.0 + s->loc.row*DY*NY;

    s->mpi_wait_time = 0.0;

    init_neighbors(s);

    /* allocate memory for event masks */

    s->events = (eventmask*) malloc( (1 + s->n_neighbors)*sizeof(eventmask));
    if ( s->events == NULL ) {
      fprintf(stderr, "ERROR:init_state: malloc of s->events failed\n");
      exit(1);
    }

    s->event_store = (unsigned char*) malloc( nsteps * N/8 );
    if ( s->event_store == NULL ) {
      fprintf(stderr, "ERROR:init_state: malloc s->event_store failed\n");
      exit(1);
    }

    init_state_ppu(s);

    sprintf(filename, "input_%d", s->comm_id);
    pv_output(filename, 0.2, s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, s->I);

    if (DEBUG) printf("[%d] init_state: local eventmask is %p, n_neighbors is %d\n",
		      s->comm_id, s->events, s->n_neighbors);
    return 0;
}


#ifdef OLD_VERSION
int update_partial_state(PVState* s, int conn_id)
{
    int i, sum = 0;
    eventmask* emk = column_event(s, conn_id);

    /* for now, just sum up the number of synapses that fired */
    for (i = 0; i < NUM_MASK_EVENTS; i++) {
        sum += emk->event[i];	// TODO - do bit manipulations
    }
    if (DEBUG) {
      printf("[%d] update_state: conn eventmask is %p, conn_id is %d, comm_id is %d, sum is %d\n",
	     s->comm_id, emk->event, conn_id, s->neighbors[conn_id], sum);
    }

    return 0;
}
#endif


int pv_row(PVState* s, int comm_id)
{
    return (int) comm_id/s->n_cols;
}


int pv_col(PVState* s, int comm_id)
{
  return ( comm_id - s->n_cols * pv_row(s, comm_id) );
}


/**
 * Returns the comm_id of the northwestern HyperColumn
 */
int pv_northwest(PVState* s, int comm_id)
{
    if (hasNorthernNeighbor(s, comm_id) == 0) return -1;
    return pv_west(s, comm_id + s->n_cols);
}


/**
 * Returns the comm_id of the northern HyperColumn
 */
int pv_north(PVState* s, int comm_id)
{
    if (hasNorthernNeighbor(s, comm_id) == 0) return -1;
    return (comm_id + s->n_cols);
}


/**
 * Returns the comm_id of the northeastern HyperColumn
 */
int pv_northeast(PVState* s, int comm_id)
{
    if (hasNorthernNeighbor(s, comm_id) == 0) return -1;
    return pv_east(s, comm_id + s->n_cols);
}


/**
 * Returns the comm_id of the western HyperColumn
 */
int pv_west(PVState* s, int comm_id)
{
    if (hasWesternNeighbor(s, comm_id) == 0) return -1;
    return (pv_row(s, comm_id)*s->n_cols + ((comm_id - 1) % s->n_cols));
}


/**
 * Returns the comm_id of the eastern HyperColumn
 */
int pv_east(PVState* s, int comm_id)
{
    if (hasEasternNeighbor(s, comm_id) == 0) return -1;
    return (pv_row(s, comm_id)*s->n_cols + ((comm_id + 1) % s->n_cols));
}


/**
 * Returns the comm_id of the southwestern HyperColumn
 */
int pv_southwest(PVState* s, int comm_id)
{
    if (hasSouthernNeighbor(s, comm_id) == 0) return -1;
    return pv_west(s, comm_id - s->n_cols);
}


/**
 * Returns the comm_id of the southern HyperColumn
 */
int pv_south(PVState* s, int comm_id)
{
    if (hasSouthernNeighbor(s, comm_id) == 0) return -1;
    return (comm_id - s->n_cols);
}


/**
 * Returns the comm_id of the southeastern HyperColumn
 */
int pv_southeast(PVState* s, int comm_id)
{
    if (hasSouthernNeighbor(s, comm_id) == 0) return -1;
    return pv_east(s, comm_id - s->n_cols);
}


/**
 * Returns the sender rank for the given connection index
 */
int neighbor_index(PVState* s, int comm_id, int index)
{
    switch (index) {
      case 0: /* northwest */
	return pv_northwest(s, comm_id);
      case 1: /* north */
	return pv_north(s, comm_id);
      case 2: /* northeast */
	return pv_northeast(s, comm_id);
      case 3: /* west */
	return pv_west(s, comm_id);
      case 4: /* east */
	return pv_east(s, comm_id);
      case 5: /* southwest */
	return pv_southwest(s, comm_id);
      case 6: /* south */
	return pv_south(s, comm_id);
      case 7: /* southeast */
	return pv_southeast(s, comm_id);
      default:
	fprintf(stderr, "ERROR:neighbor_index: bad index\n");
    }	
    return -1;
}


eventmask* column_event(PVState* s, int conn_id)
{
    return &(s->events[conn_id+1]);
}


int output_partial_state(PVState* s, int time_step)
{
    int i;
    char filename[64];
    float phimax = -100000.0;
    float* phi = s->phi;

    //sprintf(filename, "phi%d", time_step);
    sprintf(filename, "phi");
    for (i = 0; i < N; i++) {
      if (phimax < phi[i]) phimax = phi[i];
    }
    
    pv_output(filename, phimax/2., s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, phi);
    pv_output_on_circle(time_step, "phi", 1.0, phi);
    
    return 0;
}


void output_state(PVState* s, int time_step)
{
    int i;
    char filename[64];
    float fave = 0.0;
    float Vave = 0.0;
/*     float Vmax = -100000.0; */
    float phimax = -100000.0;

    float* phi = s->phi;
    float* f = s->events[0].event;
    float* V = s->V;
    float* h = s->h;
    float* H = s->H;

    /* save event mask */

    int offset = time_step * N/8;
    compress_float_mask(N, f, &s->event_store[offset]);

    /* graphics output */

    for (i = 0; i < N; i++) {
      if (phimax < phi[i]) phimax = phi[i];
    }
    for (i = 0; i < N; i++) {
      Vave += V[i];
    }
    for (i = 0; i < N; i++) {
      fave += f[i];
    }
    
    pv_output_on_circle(time_step, "V  ", 0.6, V);
    pv_output_events_on_circle(time_step, f, h);

    sprintf(filename, "f%d_%d", time_step, s->comm_id);
    //sprintf(filename, "f%d", s->comm_id);
    pv_output(filename, 0.5, s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, f);

    sprintf(filename, "V%d_%d", time_step, s->comm_id);
    //sprintf(filename, "V%d", s->comm_id);
    pv_output(filename, -1000., s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, V);

    sprintf(filename, "h%d_%d", time_step, s->comm_id);
    //sprintf(filename, "h%d", s->comm_id);
    pv_output(filename, 0.5, s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, h);

    sprintf(filename, "Vinh%d_%d", time_step, s->comm_id);
    //sprintf(filename, "Vinh%d", s->comm_id);
    pv_output(filename, -1000., s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, H);

/*     //    sprintf(filename, "phi%d_%d", time_loop, s->comm_id); */
/*     sprintf(filename, "./output/phi%d", s->comm_id); */
/*     pv_output(filename, -1000., s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, phi); */

    printf("loop=%d:  fave=%f, Vave=%f\n", time_step, 1000*fave/N, Vave/N);
}


int output_final_state(PVState* s, int nsteps)
{
    int i;
    size_t count, size;
    char filename[64];
    float fmax = 1.0;
    unsigned char* recv_buf;
    FILE* fp = NULL;

    float* f = s->events[0].event;

    //    sprintf(filename, "f%d_%d", time_loop, s->comm_id);
    sprintf(filename, "f%d", s->comm_id);
    pv_output(filename, fmax/2., s->loc.x0, s->loc.y0, s->loc.x, s->loc.y, s->loc.o, f);

    /* gather event masks and dump them one time step at a time */

    size = N/8;

    if (s->comm_id == 0) {
      sprintf(filename, "%s/events.bin", OUTPUT_PATH);
      fp = fopen(filename, "w");
      if (fp == NULL) {
    	  fprintf(stderr, "ERROR:output_final_state: error opening events.bin\n");
    	  return 1;
      }

      /* allocate memory for ALL processors (one bit per event) */
      recv_buf = (unsigned char*) malloc( size*s->comm_size );
      if (recv_buf == NULL) {
    	  fprintf(stderr, "ERROR:output_final_state: error malloc of output buffer of size %ld\n", size);
    	  return 1;
      }
    }

    for (i = 0; i < nsteps; i++) {
      if (DEBUG) {
    	  printf("[%d] output_final_state: gather of size %d\n", s->comm_id, (int) size);
      }
     // MPI_Gather(&s->event_store[i*size], size, MPI_CHAR, recv_buf, size, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (s->comm_id == 0) {
    	  count = fwrite(recv_buf, sizeof(unsigned char), size*s->comm_size, fp);
    	  if (count != size*s->comm_size) {
    		  fprintf(stderr, "ERROR:output_final_state: error writing output buffer of size %ld\n",
		              size*s->comm_size);
    	  }
      }
    }

    if (s->comm_id == 0) {
      fclose(fp);
    }
    return 0;
}


void error_msg(PVState* s, int err, char* loc)
{
    if (err) {
        fprintf(stderr, "[%d] %s: ERROR(=%d) occurred in ???\n", s->comm_id, loc, err);
    }
}
