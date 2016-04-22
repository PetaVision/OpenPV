#ifndef PV_H_
#define PV_H_

// This file doesn't appear to be used and much of it is (should be)
// deprecated anyway
//

#define DEBUG		0	/* turns on debugging output */
/******************************************************************/
/*FILE DEFINITIONS                                                */
/******************************************************************/

#undef  OUTPUT_PS		/* output as postscript file */
#define OUTPUT_BIN		/* output as binary file */
#define INPUT_PATH  "./io/input/circle1_input.bin"   /*name of the input file*/
#define PARAMS_FILE "params.txt" // log runtime parameters

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
#define NOISE_AMP       (1.0*0.5*V_TH_0/DT_d_TAU) // maximum amplitude of noise if present
#define NOISE_FREQ      1.0          //0.5                // prob of noise input on each time step
#define MIN_V                   -4.0*V_TH_0                   //minimum potential
/*****************************************************************************************************/

//Inhibition basics
//#define INHIBIT_ON                                           //inhibition flag, define to turn on inhibition
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
#define COCIRC_SCALE	 0.33*(V_TH_0 / DT_d_TAU / (1 - INHIB_FRACTION))  //Scale for Excite to excite cells
//#define COCIRC_SCALE	 (0.5*V_TH_0/DT_d_TAU)	                           // (.025,0)->stable, (.05,0)->unstable
#define EXCITE_R2        20*(NX*NX+NY*NY)                                  //cut-off radius for excititory cells(infinite wrt the screen size for now)
#define INHIB_FRACTION   0.9                                               // fraction of inhibitory connections
#define INHIBIT_SCALE	 1.0	                                           // reduce inhibition (w < 0) by this amount
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

#endif /* PV_H_ */
