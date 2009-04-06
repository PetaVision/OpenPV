/*
 * zucker.h
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

#ifndef ZUCKER_H_
#define ZUCKER_H_

// Tuning parameters.
// TODO: Not sure if they should be more global/shared,
// since they are standard LIF parameters, but we might
// want different value for different layers.
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

/****************************************************************/
/*PHI PARAMETERS- for each connection (all follow first scheme) */
/****************************************************************/
// E->E
//Excite to Excite connection
#define SIG_C_D_x2	 (2*8.0*8.0)	                                   // (squared and times 2)
#define SIG_C_P_x2	 (2*1*DTH*DTH)
#define COCIRC_SCALE	 0.33*(V_TH_0 / DT_d_TAU / (1 - INHIB_FRACTION))  //Scale for Excite to excite cells
//#define COCIRC_SCALE	 (0.5*V_TH_0/DT_d_TAU)	                           // (.025,0)->stable, (.05,0)->unstable
#define EXCITE_R2        20*(NX*NX+NY*NY)                                  //cut-off radius for excititory cells(infinite wrt the screen size for now)
#define INHIB_FRACTION   0.9                                               // fraction of inhibitory connections
#define INHIBIT_SCALE	 1.0	                                           // reduce inhibition (w < 0) by this amount
/**********************************************************************************************************/

#endif /* ZUCKER_H_ */
