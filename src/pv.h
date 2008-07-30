#ifndef PV_H_
#define PV_H_

#define DEBUG		0	/* turns on debugging output */
/******************************************************************/
/*FILE DEFINITIONS                                                */
/******************************************************************/

#undef OUTPUT_PS		/* output as postscript file */
#define OUTPUT_BIN		/* output as binary file */
#define OUTPUT_PATH "./output"	/* location of the output files */
#define INPUT_PATH  "./io/input/circle1_input.bin"   /*name of the input file*/
#define PARAMS_FILE "params.txt" // log runtime parameters

/*******************************************************************/
/*BASIC PARAMERS (SIZE OF MODEL)                                   */
/*******************************************************************/

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
#define MIN_DX 1.e-8
#define DX    1.0			/* pixel units, original 80.0/10. */
#define DY    1.0			/* pixel units, origianl 80.0/10. */
#define DTH   (180.0/NO)	/* degrees */

#define MS_PER_TIMESTEP		1.0
/*****************************************************************************************************/

/*******************************************************/ 
/*  Poissonian Input                                   */
/*******************************************************/
#define POISSON_INPUT		0			// 0=off: continuous spikes. 1=on, Poisson
#define POISSON_INPUT_RATE	1000.0	 		// Firing rate of "on" edges in Hz
#define POISSON_RATE_TIMESTEPS	((POISSON_INPUT_RATE*MS_PER_TIMESTEP)/1000.0)

/*******************************************************/ 
/*MEMBRANE POTENTIAL PARAMETERS                        */
/*******************************************************/

//Excitation basics
#define V_TH_0			0.5			  // threshold potential
#define DT_d_TAU		0.125		          // rate of change of excitation
#define ALPHA			0.01		          // desired fraction active per time step
#define NOISE_AMP               (1.0*0.5*V_TH_0/DT_d_TAU) // maximum amplitude of noise if present
#define NOISE_FREQ              1.0  //0.5                // prob of noise input on each time step
#define MIN_V                   -4.0*V_TH_0                   //minimum potential
/*****************************************************************************************************/

//Inhibition basics
#define INHIBIT_ON                                           //inhibition flag, define to turn on inhibition
#define DT_d_TAU_INH	        0.05 // (DT_d_TAU/2)  // rate of change of inhibition
#define V_TH_0_INH              1.0*V_TH_0           //  threshold potential for inhibitory neurons
#define NOISE_AMP_INH           NOISE_AMP                    // maximum amplitude of noise if present
#define NOISE_FREQ_INH          NOISE_FREQ                   // prob of noise input on each time step
#define MIN_H                   -V_TH_0_INH                  //minimum inhibitory potential
/*****************************************************************************************************/


/****************************************************************/
/*PHI PARAMETERS- for each connection (all follow first scheme) */ 
/****************************************************************/

//Excite to Excite connection
#define SIG_C_D_x2	 (2*8.0*8.0)	                                   // (squared and times 2)
#define SIG_C_P_x2	 (2*1*DTH*DTH)
#define COCIRC_SCALE	 0.033*(V_TH_0 / DT_d_TAU / (1 - INHIB_FRACTION))  //Scale for Excite to excite cells
//#define COCIRC_SCALE	 (0.5*V_TH_0/DT_d_TAU)	                           // (.025,0)->stable, (.05,0)->unstable
#define EXCITE_R2        20*(NX*NX+NY*NY)                                  //cut-off radius for excititory cells(infinite wrt the screen size for now)
#define INHIB_FRACTION   0.9                                               // fraction of inhibitory connections
#define INHIBIT_SCALE	 0*1.0	                                           // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

//Inhibit to Excite connection
#define INHIB_DELAY       3                   //number of times steps delay (x times slower than excititory conections)
#define SIG_I_D_x2        (2*2.0*2.0)         // sigma (square and time 2) for inhibition to exicititory connections 
#define SIG_I_P_x2        (2*1.0*DTH*DTH)   
#define INHIB_R2          9*(DX*DX+DY*DY)     //square of radius of inhibition
#define SCALE_INH         -100.0*COCIRC_SCALE
#define INHIB_FRACTION_I  0.8                 // fraction of inhibitory connections
#define INHIBIT_SCALE_I	  0*1.0	              // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

//Gap Junctions
#define SIG_G_D_x2        (2*2.0*2.0)           //sigma (square and times 2) for gap junctions (inhibit to inhibit)
#define SIG_G_P_x2        (2*1.0*DTH*DTH)
#define GAP_R2            4*(DX*DX+DY*DY)         //square of radius of gap junctions keep small
#define SCALE_GAP         4.0*COCIRC_SCALE
#define INHIB_FRACTION_G  0.9                   // fraction of inhibitory connections
#define INHIBIT_SCALE_G	  0*1.0	                // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/
 
//Excite to Inhibit connection
#define SIG_E2I_D_x2        SIG_C_D_x2
#define SIG_E2I_P_x2        SIG_C_P_x2
#define E2I_R2              EXCITE_R2
#define E_TO_I_SCALE        4.0*COCIRC_SCALE
#define INHIB_FRACTION_E2I  0.8             // fraction of onhibitory connections
#define INHIBIT_SCALE_E2I   0*1.0	    // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

/*****************************************************************/
/*MISCELLANEOUS-(Image and other basic definitions)              */
/*****************************************************************/

// Image basics
#define I_MAX       (1.0*0.5*V_TH_0/DT_d_TAU) // maximum image intensity
#define CLUTTER_PROB     0.01            // prob of clutter in image


//Basic definitons
#define PI				3.1415926535897931

#define RAD_TO_DEG_x2	(2.0*180.0/PI)

#define NUM_NEIGHBORS   8	/* number of neighboring HyperColumns */

#define eventtype_t     float		/* data type for an event */
#define EVENT_TYPE_MPI  MPI_FLOAT	/* MUST BE SAME AS ABOVE */

#define NUM_MASK_EVENTS	N	/* number of elements in event mask */
#define SEED 1

#define MAX_FILENAME 128


/**********************************************************************/

/**
 * data type for event mask (floats for now, could be compressed to bits)
 */
// TODO - number of neurons per layer will have to change, redo this
typedef struct eventmask_ {
    eventtype_t event[NUM_MASK_EVENTS];
} eventmask;
/**************************************************************************/

/*********/
/* API's */
/*********/
int comm_init(int* pargc, char*** pargv, int* rank, int* size);
int comm_finalize();

//int recver_rank(PVState* s, int id);
//int sender_rank(PVState* s, int id);

void pv_output(char* path, float threshold, float x0, float y0,
	       float x[], float y[], float o[], float I[]);

void compress_float_mask(int size, float buf[], unsigned char bits[]);

void post(float threshold, float x0, float y0, float x[], float y[], float th[], float F[]);

#endif /* PV_H_ */
