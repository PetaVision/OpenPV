/*
 * default_params.h
 *
 *  Created on: Dec 13, 2008
 *      Author: rasmussn
 */

#ifndef DEFAULT_PARAMS_H_
#define DEFAULT_PARAMS_H_

#define NOISE_AMP 0.25
#define V_REST -70.0
#define V_EXC 0.0
#define V_INH -75.0
#define V_INHB -90.0
#define VTH_REST (V_REST + 15.0)
#define TAU_VMEM 20.0
#define TAU_EXC 1.0
#define TAU_INH 5.0
#define TAU_INHB 10.0
#define DELTA_G_INHB 1.0
#define TAU_RATE 100000.0 // in ms
#define TAU_WMAX 100000.0 // in ms
#define TAU_VTH 10.0
#define TAU_VTHREST 10000.0 // in ms
#define DELTA_VTH 5.0
#define DELTA_VTH_REST 0.1
#define GLOBAL_GAIN 1.0
#define ASPECT_RATIO 4.0
#define RMAX_EDGE 4.0
#define SIGMA_EDGE 2.0
#define RMAX_COCIRC 24.0
#define SIGMA_DIST_COCIRC 48.0
#define RMAX_FEEDBACK 4.0
#define SIGMA_DIST_FEEDBACK 2.0
#define EXCITE_DELAY 0
#define INHIB_DELAY 0
#define EXCITE_VEL 999999.0 // 2*RMAX
#define INHIB_VEL 999999.0

// estimate of maximum activity in a local layer
#define EST_MAX_ACTIVITY 256

// Sigmoid layer inverse output flag
#define INVERSEFLAG 0
#define SIGMOIDFLAG 1
#define SIGMOIDALPHA 0.1
// weight parameters
#define WEIGHT_DEAD_VALUE 0.0
#define WEIGHT_MIN_VALUE 0.001

#endif /* DEFAULT_PARAMS_H_ */
