#ifndef PV_H_
#define PV_H_

#define DEBUG		0	/* turns on debugging output */
#undef OUTPUT_PS		/* output as postscript file */
#define OUTPUT_BIN		/* output as binary file */

#define OUTPUT_PATH "./output"	/* location of the output files */
#define INPUT_PATH  "./io/input/circle1_input.bin"   /*name of the input file*/

#define NUM_SPUS	8

#define MAX_LAYERS      2

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

#define SIG_C_D_x2	(2*8.0*8.0)	// (squared and times 2)
#define SIG_C_P_x2	(2*1*DTH*DTH)

#define V_TH_0			0.5			// threshold potential
#define V_TH_0_INH      0.5         // threshold potential for inhibitory neurons
#define DT_d_TAU		0.125		// rate of change of excitation
#define DT_d_TAU_INH	0.05 // (DT_d_TAU/2)// rate of change of inhibition
#define ALPHA			0.01		// desired fraction active per time step
#define INHIB_FRACTION  0.9             // fraction of inhibitory connections
#define INHIBIT_SCALE	1.0			// reduce inhibition (w < 0) by this amount
#define COCIRC_SCALE	0.33*(V_TH_0 / DT_d_TAU / (1 - INHIB_FRACTION))	
//#define COCIRC_SCALE	(0.5*V_TH_0/DT_d_TAU)	// (.025,0)->stable, (.05,0)->unstable

#define NOISE_AMP       (1.0*0.5*V_TH_0/DT_d_TAU) // maximum amplitude of noise if present
#define NOISE_FREQ      1.0  //0.5                    // prob of noise input on each time step

#define NOISE_AMP_INH   NOISE_AMP   // maximum amplitude of noise if present
#define NOISE_FREQ_INH  NOISE_FREQ  // prob of noise input on each time step
#define INHIBIT_AMP     10.0           // amplitude of inhibitory input


#define INHIB_R2        4            //square of radius of inhibition to be used later
#define INHIB_FRACTION        0.9

#define I_MAX       (1.0*0.5*V_TH_0/DT_d_TAU) // maximum image intensity
#define CLUTTER_PROB     0.01            // prob of clutter in image

#define PI				3.1415926535897931

#define RAD_TO_DEG_x2	(2.0*180.0/PI)

#define NUM_NEIGHBORS   8	/* number of neighboring HyperColumns */

#define eventtype_t     float		/* data type for an event */
#define EVENT_TYPE_MPI  MPI_FLOAT	/* MUST BE SAME AS ABOVE */

#define NUM_MASK_EVENTS	N	/* number of elements in event mask */
#define SEED 1

#define MAX_FILENAME 128


/**
 * data type for event mask (floats for now, could be compressed to bits)
 */
// TODO - number of neurons per layer will have to change, redo this
typedef struct eventmask_ {
    eventtype_t event[NUM_MASK_EVENTS];
} eventmask;


/* API's */

int comm_init(int* pargc, char*** pargv, int* rank, int* size);
int comm_finalize();

//int recver_rank(PVState* s, int id);
//int sender_rank(PVState* s, int id);

void pv_output(char* path, float threshold, float x0, float y0,
	       float x[], float y[], float o[], float I[]);

void compress_float_mask(int size, float buf[], unsigned char bits[]);

void post(float threshold, float x0, float y0, float x[], float y[], float th[], float F[]);

#endif /* PV_H_ */
