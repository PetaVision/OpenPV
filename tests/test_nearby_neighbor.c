#include "../src/layers/elementals.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
   int kPre, kPost;
   int min = -10000;   
   int max =  10000;   

   int scale = 0;

   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scale);
      if (kPost != kPre) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scale=%d, kPre=%d, kPost=%d\n", scale, kPre, kPost);
         exit(1);
      }
   }

   scale = 1;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scale);
      if (kPost != kPre/2) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scale=%d, kPre=%d, kPost=%d\n", scale, kPre, kPost);
         exit(1);
      }
   }

   scale = 2;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scale);
      if (kPost != kPre/4) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scale=%d, kPre=%d, kPost=%d\n", scale, kPre, kPost);
         exit(1);
      }
   }

   scale = -1;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scale);
      if (kPost != 2*kPre) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scale=%d, kPre=%d, kPost=%d\n", scale, kPre, kPost);
         exit(1);
      }
   }

   scale = -2;
   for (kPre = min; kPre < max; kPre++) {
      kPost = nearby_neighbor(kPre, scale);
      if (kPost != 4*kPre) {
         printf("FAILED:TEST_NEARBY_NEIGHBOR: scale=%d, kPre=%d, kPost=%d\n", scale, kPre, kPost);
         exit(1);
      }
   }

   return 0;
}
