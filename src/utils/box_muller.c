/*
 * box_muller.c
 *
 *  Created on: Feb 12, 2010
 *      Author: manghel
 */

/* boxmuller.c           Implements the Polar form of the Box-Muller
                         Transformation

                      (c) Copyright 1994, Everett F. Carter Jr.
                          Permission is granted by the author to use
                          this software for any application provided this
                          copyright notice is preserved.

*/
#ifdef OBSOLETE // Marked obsolete Aug 28, 2013.  Use the Random class

#include "pv_random.h"
#include <math.h>

float box_muller(float m, float s)      /* normal random variate generator */
{                                       /* mean m, standard deviation s */
   float x1, x2, w, y1;
   static float y2;
   static int use_last = 0;

   if (use_last)                   /* use value from previous call */
   {
      y1 = y2;
      use_last = 0;
   }
   else
   {
      do {
         x1 = 2.0 * pv_random_prob() - 1.0;
         x2 = 2.0 * pv_random_prob() - 1.0;
         w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );

      w = sqrt( (-2.0 * log( w ) ) / w );
      y1 = x1 * w;
      y2 = x2 * w;
      use_last = 1;
   }

   return( m + y1 * s );
}
#endif // OBSOLETE
