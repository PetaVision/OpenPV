/*
 * neural_tuning.h
 *
 *  Created on: Aug 6, 2008
 *      Author: dcoates
 */

#ifndef NEURAL_TUNING_H_
#define NEURAL_TUNING_H_

// Neural tuning parameters
//

#define MIN_DX 1.e-8
#define DX    1.0		/* pixel units, original 80.0/10. */
#define DY    1.0		/* pixel units, origianl 80.0/10. */
#define DTH   (180.0/NO)	/* degrees */
#define K_0              0.0                   //  curvature start at zero (straight line)
//#define DK              (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature
#define SIG_C_K_x2       2.0*DK*DK        // tolerance from target curvature:l->kappa
#define MIN_DENOM        1.0E-10               // 1.0E-10;
#define MS_PER_TIMESTEP		1.0
/******************************************************************************************************/

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
//#define NOISE_AMP             (0.1*V_TH_0/DT_d_TAU) // maximum amplitude of noise if present
#define NOISE_FREQ            0.5  //0.5                // prob of noise input on each time step
#define MIN_V                   -4.0*V_TH_0               //minimum potential
/*****************************************************************************************************/

//Inhibition basics
#define INHIBIT_ON                                      //inhibition flag, define to turn on inhibition
#define DT_d_TAU_INH	        0.065  // (DT_d_TAU/2)    // rate of change of inhibition
#define V_TH_0_INH              1.0*V_TH_0              //  threshold potential for inhibitory neurons
#define NOISE_AMP_INH          NOISE_AMP               // maximum amplitude of noise if present
#define NOISE_FREQ_INH          NOISE_FREQ              // prob of noise input on each time step
#define MIN_H                   -1.0*V_TH_0_INH             //minimum inhibitory potential
/*****************************************************************************************************/



/****************************************************************/
/*PHI PARAMETERS- for each connection (all follow first scheme) */
/****************************************************************/

#define WEIGHT_SCALE  10.0   //0.033*(V_TH_0 / DT_d_TAU / (1 - INHIB_FRACTION))

//Excite to Excite connection
#define SIG_C_D_x2	 (2*4.0*4.0)	                                   // (squared and times 2)
#define SIG_C_P_x2	 (2*1.0*DTH*DTH)
#define COCIRC_SCALE	 (1.0*WEIGHT_SCALE)                                     // Scale for Excite to excite cells
#define EXCITE_R2        8*8*(DX*DX+DY*DY)                                  // cut-off radius for excititory cells(infinite wrt the screen size for now)
#define INHIB_FRACTION   0.9                                               // fraction of inhibitory connections
#define INHIBIT_SCALE	 0*1.0	                                           // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

//Inhibit to Excite connection
//#define INHIB_DELAY       3                      //number of times steps delay (x times slower than excititory conections)
#define SIG_I_D_x2        (2*2.0*2.0)            // sigma (square and time 2) for inhibition to exicititory connections
#define SIG_I_P_x2        (2*1.0*DTH*DTH)
#define INHIB_R2          4.0*4.0*(DX*DX+DY*DY)  //square of radius of inhibition
#define SCALE_INH        (-125.0*WEIGHT_SCALE)
#define INHIB_FRACTION_I  0.8                    // fraction of inhibitory connections
#define INHIBIT_SCALE_I	  0*1.0	                 // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

//Inhibition of the inhibition
#define SIG_II_D_x2        SIG_I_D_x2          //sigma (square and time 2) for inhibition to exicititory connections
#define SIG_II_P_x2        SIG_I_P_x2
#define INHIBI_R2          INHIB_R2            //square of radius of inhibition
#define SCALE_INHI         (-10.0*WEIGHT_SCALE)
#define INHIB_FRACTION_II  0.8                    // fraction of inhibitory connections
#define INHIBIT_SCALE_II   0*1.0	                 // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

//Gap Junctions
#define SIG_G_D_x2        (2*2.0*2.0)           //sigma (square and times 2) for gap junctions (inhibit to inhibit)
#define SIG_G_P_x2        (2*1.0*DTH*DTH)
#define GAP_R2            4.0*4.0*(DX*DX+DY*DY)         //square of radius of gap junctions keep small
#define SCALE_GAP        2.0*WEIGHT_SCALE
#define INHIB_FRACTION_G  0.9                   // fraction of inhibitory connections
#define INHIBIT_SCALE_G	  0*1.0	                // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

//Excite to Inhibit connection
#define SIG_E2I_D_x2        SIG_C_D_x2
#define SIG_E2I_P_x2        SIG_C_P_x2
#define E2I_R2              EXCITE_R2
#define E_TO_I_SCALE        6.0*WEIGHT_SCALE
#define INHIB_FRACTION_E2I  0.9             // fraction of onhibitory connections
#define INHIBIT_SCALE_E2I   0*1.0	    // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

// Others:
#define I_MAX       1.0*(1.0*0.5*V_TH_0/DT_d_TAU) // maximum image intensity
#define CLUTTER_PROB     0.01            // prob of clutter in image

#endif /* NEURAL_TUNING_H_ */
