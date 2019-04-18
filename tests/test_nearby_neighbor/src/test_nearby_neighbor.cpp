#include "utils/PVLog.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <utils/conversions.h>

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
 *  ambiguous and the neuron to the left is chosen.  If the density of the
 *  post-synaptic layer decreases, there is no ambiguity.
 *
 */

int main(int argc, char *argv[]) {
   int kPre, kPost, kBack, a, ans, test;

   // must start at a negative odd number
   //
   int min = -100000;
   int max = 100000;

   // a < 0 tests assum start is some factor of 8 (minus 1)
   min = -1 - 8 * 10000;
   max = 8 * 10000;

   int log2ScaleDiff;

   // post-synaptic layer has same size
   //

   a             = 1;
   log2ScaleDiff = 0;
   ans           = a * min + 0;
   test          = 1;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, log2ScaleDiff);
      if (kPost != kPre) {
         Fatal().printf(
               "FAILED:TEST_NEARBY_NEIGHBOR: "
               "test==%d log2ScaleDiff==%d kPre==%d kPost==%d ans==%d\n",
               test,
               log2ScaleDiff,
               kPre,
               kPost,
               ans);
      }
      ans += 1;
   }

   // post-synaptic layer density decreases by 2 (dx increases by 2)
   //
   log2ScaleDiff = 1;
   ans           = (min - 1) / 2;
   test          = 2;
   for (kPre = min; kPre <= max; kPre++) {
      kPost = nearby_neighbor(kPre, log2ScaleDiff);
      if (kPost != ans) {
         Fatal().printf(
               "FAILED:TEST_NEARBY_NEIGHBOR: "
               "test==%d log2ScaleDiff==%d kPre==%d kPost==%d ans==%d\n",
               test,
               log2ScaleDiff,
               kPre,
               kPost,
               ans);
      }
      if ((kPre - min + 1) % 2 == 1)
         ans += 1;
   }

   // post-synaptic layer density decreases by 4 (dx increases by 4)
   //
   log2ScaleDiff = 2;
   ans           = (min - 3) / 4;
   test          = 3;
   for (kPre = min; kPre <= max; kPre++) {
      kPost = nearby_neighbor(kPre, log2ScaleDiff);
      if (kPost != ans) {
         Fatal().printf(
               "FAILED:TEST_NEARBY_NEIGHBOR: "
               "test==%d log2ScaleDiff==%d kPre==%d kPost==%d ans==%d\n",
               test,

               log2ScaleDiff,
               kPre,
               kPost,
               ans);
      }
      if ((kPre - min + 1) % 4 == 1)
         ans += 1;
   }

   // post-synaptic layer density decreases by 8 (dx increases by 8)
   //
   log2ScaleDiff = 3;
   ans           = (min - 7) / 8;
   test          = 4;
   for (kPre = min; kPre <= max; kPre++) {
      kPost = nearby_neighbor(kPre, log2ScaleDiff);
      if (kPost != ans) {
         Fatal().printf(
               "FAILED:TEST_NEARBY_NEIGHBOR: "
               "test==%d log2ScaleDiff==%d kPre==%d kPost==%d ans==%d\n",
               test,

               log2ScaleDiff,
               kPre,
               kPost,
               ans);
      }
      if ((kPre - min + 1) % 8 == 1)
         ans += 1;
   }

   // post-synaptic layer density increases by 2 (dx decreases by 2)
   //
   a             = 2;
   log2ScaleDiff = -1;
   ans           = a * min + 0;
   test          = 5;
   for (kPre = min; kPre <= max; kPre++) {
      kPost = nearby_neighbor(kPre, log2ScaleDiff);
      kBack = nearby_neighbor(kPost, -log2ScaleDiff);
      if (kPost != ans && kBack != kPre) {
         Fatal().printf(
               "FAILED:TEST_NEARBY_NEIGHBOR: "
               "test==%d log2ScaleDiff==%d kPre==%d kPost==%d kBack==%d ans==%d\n",
               test,

               log2ScaleDiff,
               kPre,
               kPost,
               kBack,
               ans);
      }
      ans += a;
   }

   // post-synaptic layer density increases by 4 (dx decreases by 4)
   //
   a             = 4;
   log2ScaleDiff = -2;
   ans           = a * min + 1;
   test          = 6;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, log2ScaleDiff);
      kBack = nearby_neighbor(kPost, -log2ScaleDiff);
      if (kPost != ans && kBack != kPre) {
         Fatal().printf(
               "FAILED:TEST_NEARBY_NEIGHBOR: "
               "test==%d log2ScaleDiff==%d kPre==%d kPost==%d kBack==%d ans==%d\n",
               test,

               log2ScaleDiff,
               kPre,
               kPost,
               kBack,
               ans);
      }
      ans += a;
   }

   // post-synaptic layer density increases by 8 (dx decreases by 8)
   //
   a             = 8;
   log2ScaleDiff = -3;
   ans           = a * min + 3;
   test          = 7;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, log2ScaleDiff);
      kBack = nearby_neighbor(kPost, -log2ScaleDiff);
      if (kPost != ans && kBack != kPre) {
         Fatal().printf(
               "FAILED:TEST_NEARBY_NEIGHBOR: "
               "test==%d log2ScaleDiff==%d kPre==%d kPost==%d kBack==%d ans==%d\n",
               test,

               log2ScaleDiff,
               kPre,
               kPost,
               kBack,
               ans);
      }
      ans += a;
   }

   return 0;
}
