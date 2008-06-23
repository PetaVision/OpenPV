#ifndef PV_H_
#define PV_H_

#include <mpi.h>

#define DEBUG		0	/* turns on debugging output */
#define OUTPUT_PS		/* output as postscript file */
#undef OUTPUT_BIN		/* output as binary file */

#define OUTPUT_PATH "./output"	/* location of the output files */

#define NUM_SPUS	8

#define NX    36			/* 48  */
#define NY    36			/* 48  */
#define NO     8			/* 16  */
#define N     (NX*NY*NO)	/* 36864 */
#define NSPU  (N/NUM_SPUS)	/*  */

//#define CHUNK_SIZE   1024	/* 81*1024 = 82944 */
/* in optimized version, must be divisible by NO */
/* and nicely sized in terms of cache lines */
#define CHUNK_SIZE   N		/* N/NO */

/* weight parameters */

#define MIN_DX 1.e-8

#define DX    1.0			/* pixel units, original 80.0/10. */
#define DY    1.0			/* pixel units, origianl 80.0/10. */
#define DTH   (180.0/NO)	/* degrees */

#define SIG_C_D_x2	(2*9.0*9.0)	// (squared and times 2)
#define SIG_C_P_x2	(2*1*DTH*DTH)

#define V_TH_0			0.5			// threshold potential
#define V_TH_0_INH      0.5         // threshold potential for inhibitory neurons
#define DT_d_TAU		0.25		// rate of change of excitation
#define DT_d_TAU_INH	0.1 // (DT_d_TAU/2)// rate of change of inhibition
#define ALPHA			0.01		// desired fraction active per time step
#define INHIBIT_SCALE	1.0			// reduce inhibition (w < 0) by this amount
#define COCIRC_SCALE	(0.5*V_TH_0/DT_d_TAU)	// (.025,0)->stable, (.05,0)->unstable

#define NOISE_AMP       (1.0*0.5*V_TH_0/DT_d_TAU) // maximum amplitude of noise if present
#define NOISE_FREQ      .0001  //0.5                    // prob of noise input on each time step

#define NOISE_AMP_INH   NOISE_AMP   // maximum amplitude of noise if present
#define NOISE_FREQ_INH  NOISE_FREQ  // prob of noise input on each time step
#define INHIBIT_AMP     .1           // amplitude of inhibitory input

#define PI				3.1415926535897931
#define RAD_TO_DEG_x2	(2.0*180.0/PI)

#define NUM_NEIGHBORS   8	/* number of neighboring HyperColumns */

#define eventtype_t     float		/* data type for an event */
#define EVENT_TYPE_MPI  MPI_FLOAT	/* MUST BE SAME AS ABOVE */

#define NUM_MASK_EVENTS	N	/* number of elements in event mask */
#define SEED 1


/**
 * data type for event mask (floats for now, could be compressed to bits)
 */
typedef struct eventmask_ {
    eventtype_t event[NUM_MASK_EVENTS];
} eventmask;


typedef struct PVLocation_ {
    float	row;
    float	col;
    float	x0;
    float	y0;
    float*	x;
    float*	y;
    float*	o;
} PVLocation;


typedef struct PVState_ {
    int      comm_id;
    int      comm_size;
    int      n_rows;
    int      n_cols;
    int      n_neighbors;
    int      neighbors[NUM_NEIGHBORS];	/* mapping from neighbor index to neighbor rank */

    double   mpi_wait_time;

    float*   phi;	/* potential for partial updates */
    float*   I;		/* image */
    float*   V;		/* membrane potential */
    float*   H;		/* membrane potential for inhibitory neurons */
    float*	 h;		/* inhibition events (local only) */
    
    PVLocation  loc;
    eventmask*  events;			/* event masks */

    unsigned char* event_store;		/* storage for local bit masks */
    
} PVState;


/* API's */

int comm_init(int* pargc, char*** pargv, int* rank, int* size);
int comm_finalize();

int recver_rank(PVState* s, int id);
int sender_rank(PVState* s, int id);

int pv_row(PVState* s, int comm_id);
int pv_col(PVState* s, int comm_id);

eventmask* column_event(PVState* s, int conn_id);

void pv_init(PVState* s, int nx, int ny, int no);

int init_state(PVState* s, int comm_id, int comm_size, int n_steps);

int init_state_ppu(PVState* s);

int update_partial_state(PVState* s, int hc);

int update_state(PVState* s);

void output_state(PVState* s, int time_step);

void update_phi(int nc, int np, float phi_c[], float xc[], float yc[], float thc[],
		float xp[], float yp[], float thp[], float fp[]);

void pv_output(char* path, float threshold, float x0, float y0,
	       float x[], float y[], float o[], float I[]);

void compress_float_mask(int size, float buf[], unsigned char bits[]);

void post(float threshold, float x0, float y0, float x[], float y[], float th[], float F[]);

/* MPI communication functions */

int send_state(PVState* s, MPI_Request* req);
int recv_state(PVState* s, MPI_Request* req, int* comm_id);

/* error reporting */

void error_msg(PVState* s, int err, char* loc);

#endif /* PV_H_ */
