/*
 * rng.h
 *
 *  Created on: Feb 12, 2010
 *      Author: manghel
 */

#ifndef RNG_H_
#define RNG_H_

/* define this variable if code is to be run (or transformed to be run) on a Cell processor */
//#define CELL_BE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
   float box_muller(float,float);

#ifdef __cplusplus
}
#endif

#endif



#endif /* RNG_H_ */
