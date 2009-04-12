#include "../src/layers/elementals.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
   float dx;
   int iPre;
   int scale = 0;

   for (iPre = 0; iPre < 7; iPre++) {
      dx = deltaPosLayers(iPre, scale);
      if (dx != 0.0) {
         printf("FAILED:TEST_DELTA_POS: scale=%d, iPre=%d, dx=%f\n", scale, iPre, dx);
         exit(1);
      }
   }

   scale = 1;
   for (iPre = 0; iPre < 7; iPre++) {
      dx = deltaPosLayers(iPre, scale);
      float val = (iPre % 2) ? -0.25 : 0.25;
      if (dx != val) {
         printf("FAILED:TEST_DELTA_POS: scale=%d, iPre=%d, dx=%f\n", scale, iPre, dx);
         exit(1);
      }
   }

   scale = 2;
   for (iPre = 0; iPre < 7; iPre++) {
      dx = deltaPosLayers(iPre, scale);
      float val = 1./8.;
      if (iPre % 4 == 0) val =  3.0 * val;
      if (iPre % 4 == 1) val =  1.0 * val;
      if (iPre % 4 == 2) val = -1.0 * val;
      if (iPre % 4 == 3) val = -3.0 * val;
      if (dx != val) {
         printf("FAILED:TEST_DELTA_POS: scale=%d, iPre=%d, dx=%f\n", scale, iPre, dx);
         exit(1);
      }
   }

   return 0;
}
