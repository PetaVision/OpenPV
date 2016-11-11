#include "layers/PVLayerCube.hpp"
#include "utils/PVLog.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static int zero(float x) {
   if (fabs(x) < .00001)
      return 1;
   return 0;
}

int main(int argc, char *argv[]) {
   float dx, max;

   max = 2.0f;

   dx = deltaWithPBC(1.0f, 1.0f, max);
   if (!zero(dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (1, 1)\n");
   }

   dx = deltaWithPBC(2.0f, 1.0f, max);
   if (!zero(-1.0f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (2, 1)\n");
   }

   dx = deltaWithPBC(1.0f, 2.0f, max);
   if (!zero(1.0f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (1, 2)\n");
   }

   dx = deltaWithPBC(1.0f, 3.0f, max);
   if (!zero(2.0f - fabsf(dx))) {
      Fatal().printf("FAILED:TEST_DELTA: (1, 3)\n");
   }

   dx = deltaWithPBC(2.4f, 0.4f, max);
   if (!zero(2.0f - fabsf(dx))) {
      Fatal().printf("FAILED:TEST_DELTA: (2.4, 0.4)\n");
   }

   dx = deltaWithPBC(0.0f, 4.0f, max);
   if (!zero(0.0f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (0, 4)\n");
   }

   dx = deltaWithPBC(1.0f, 4.0f, max);
   if (!zero(-1.0f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (1, 4)\n");
   }

   dx = deltaWithPBC(3.4f, 1.0f, max);
   if (!zero(1.6f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (3.4, 1.0)\n");
   }

   max = 2.5f;

   dx = deltaWithPBC(3.5f, 1.0f, max);
   if (!zero(-2.5f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (3.5, 1.0)\n");
   }

   dx = deltaWithPBC(3.6f, 1.0f, max);
   if (!zero(2.4f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (3.6, 1.0)\n");
   }

   dx = deltaWithPBC(1.1f, 3.7f, max);
   if (!zero(-2.4f - dx)) {
      Fatal().printf("FAILED:TEST_DELTA: (1.1, 3.7)\n");
   }

   return 0;
}
