/*
 * pv_random.h
 *
 *  Created on: Apr 2, 2010
 *      Author: Craig Rasmussen
 */
#ifdef OBSOLETE // Marked obsolete Aug 28, 2013.  Use the Random class

#ifndef PV_RANDOM_H_
#define PV_RANDOM_H_

#include <stdlib.h>

#define PV_RANDOM_MAX       0x7fffffff
#define PV_INV_RANDOM_MAX   (1.0 / (double) PV_RANDOM_MAX)

#ifdef __cplusplus
extern "C"
{
#endif

#include "rng.h"

//
// random number generator functions
//

static inline void pv_srandom(unsigned long seed) {srandom(seed);}
static inline long pv_random()                    {return random();}
static inline long pv_random_max()                {return PV_RANDOM_MAX;}

static inline double pv_random_prob()
{
   return (double) pv_random() * PV_INV_RANDOM_MAX;
}

#ifdef __cplusplus
}
#endif

#endif /* PV_RANDOM_H_ */

#endif // OBSOLETE
