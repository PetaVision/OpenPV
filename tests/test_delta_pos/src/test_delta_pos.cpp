#include <utils/conversions.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils/PVLog.hpp"

int main(int argc, char* argv[])
{
   float dx;
   int kPre;
   int scale = 0;

   for (kPre = 0; kPre < 7; kPre++) {
      dx = deltaPosLayers(kPre, scale);
      if (dx != 0.0) {
         pvError().printf("FAILED:TEST_DELTA_POS: scale=%d, kPre=%d, dx=%f\n", scale, kPre, dx);
      }
   }

   scale = 1;
   for (kPre = 0; kPre < 7; kPre++) {
      dx = deltaPosLayers(kPre, scale);
      float val = (kPre % 2) ? -0.25 : 0.25;
      if (dx != val) {
         pvError().printf("FAILED:TEST_DELTA_POS: scale=%d, kPre=%d, dx=%f\n", scale, kPre, dx);
      }
   }

   scale = 2;
   for (kPre = 0; kPre < 7; kPre++) {
      dx = deltaPosLayers(kPre, scale);
      float val = 1./8.;
      if (kPre % 4 == 0) val =  3.0 * val;
      if (kPre % 4 == 1) val =  1.0 * val;
      if (kPre % 4 == 2) val = -1.0 * val;
      if (kPre % 4 == 3) val = -3.0 * val;
      if (dx != val) {
         pvError().printf("FAILED:TEST_DELTA_POS: scale=%d, kPre=%d, dx=%f\n", scale, kPre, dx);
      }
   }

   scale = -1;
   for (kPre = 0; kPre < 7; kPre++) {
      dx = deltaPosLayers(kPre, scale);
      float val = -0.5;
      if (dx != val) {
         pvError().printf("FAILED:TEST_DELTA_POS: scale=%d, kPre=%d, dx=%f\n", scale, kPre, dx);
      }
   }

   scale = -2;
   for (kPre = 0; kPre < 7; kPre++) {
      dx = deltaPosLayers(kPre, scale);
      float val = -1.5;
      if (dx != val) {
         pvError().printf("FAILED:TEST_DELTA_POS: scale=%d, kPre=%d, dx=%f\n", scale, kPre, dx);
      }
   }

   return 0;
}
