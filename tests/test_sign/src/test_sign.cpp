#include "layers/PVLayerCube.hpp"
#include "utils/PVLog.hpp"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

static int zero(float x) {
   if (fabsf(x) < 0.00001f)
      return 1;
   return 0;
}

int main(int argc, char *argv[]) {
   float s;

   s = sign(3.3f);
   if (!zero(1.0f - s)) {
      Fatal().printf("FAILED:TEST_SIGN: (3.3)\n");
   }

   s = sign(0.001f);
   if (!zero(1.0f - s)) {
      Fatal().printf("FAILED:TEST_SIGN: (.001)\n");
   }

   s = sign(-0.001f);
   if (!zero(-1.0f - s)) {
      Fatal().printf("FAILED:TEST_SIGN: (-.001)\n");
   }

   s = sign(0.0f);
   if (!zero(1.0f - s)) {
      Fatal().printf("FAILED:TEST_SIGN: (0.0)\n");
   }

   return 0;
}
