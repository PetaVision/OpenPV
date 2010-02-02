#include "../src/utils/conversions.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * The scale factor is a measure of the difference in distance scales between
 * layers.  The distance scale, dx, is the distance between neurons in
 * retinatopic units so dx in the retina is 1.
 *
 * In the PV code, the scale factor is usually stored as log base 2 of the
 * relative scale factor.  Thus if xScaleLog2 == 1 for a layer, then dx == 2
 * (the linear density of neurons is 1/2 that of the retina).  Likewise if
 * xScaleLog2 == -1, then dx == 1/2 (the linear density of the neurons is 2 times
 * that of the retina).
 *
 *  If the density of the post-synaptic layer increases, the nearby neighbor is
 *  ambiguous and the neuron to the right is chosen.  If the density of the
 *  post-synaptic layer decreases, there is no ambiguity and the nearby
 *  neighbor is just (a * kzPre) where 0 < a < 1.
 *
 */

int main(int argc, char* argv[])
{
   int kPre, kPost;
   int min = -100000;   
   int max =  100000;   

   int scaleLog2Pre  = 0;
   int scaleLog2Post = 0;

   // post-synaptic layer has same size
   //
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scaleLog2Pre, scaleLog2Post);
      if (kPost != kPre) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scaleLog2Pre==%d scaleLog2Post==%d kPre==%d kPost==%d\n",
                scaleLog2Pre, scaleLog2Post, kPre, kPost);
         exit(1);
      }
   }

   // post-synaptic layer density decreases by 2 (dx increases by 2)
   //
   scaleLog2Post = 1;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scaleLog2Pre, scaleLog2Post);
      if (kPost != kPre/2) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scaleLog2Pre==%d scaleLog2Post==%d kPre==%d kPost==%d\n",
                scaleLog2Pre, scaleLog2Post, kPre, kPost);
         exit(1);
      }
   }

   // post-synaptic layer density decreases by 4 (dx increases by 4)
   //
   scaleLog2Post = 2;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scaleLog2Pre, scaleLog2Post);
      if (kPost != kPre/4) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scaleLog2Pre==%d scaleLog2Post==%d kPre==%d kPost==%d\n",
                scaleLog2Pre, scaleLog2Post, kPre, kPost);
         exit(1);
      }
   }

   // post-synaptic layer density increases by 2 (dx decreases by 2)
   //
   scaleLog2Post = -1;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scaleLog2Pre, scaleLog2Post);
      if (kPost != (2*kPre+1)) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scaleLog2Pre==%d scaleLog2Post==%d kPre==%d kPost==%d\n",
                scaleLog2Pre, scaleLog2Post, kPre, kPost);
         exit(1);
      }
   }

   // post-synaptic layer density increases by 4 (dx decreases by 4)
   //
   scaleLog2Post = -2;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scaleLog2Pre, scaleLog2Post);
      if (kPost != (4*kPre+2)) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scaleLog2Pre==%d scaleLog2Post==%d kPre==%d kPost==%d\n",
                scaleLog2Pre, scaleLog2Post, kPre, kPost);
         exit(1);
      }
   }

   return 0;
}
