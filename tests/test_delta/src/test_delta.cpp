#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "layers/PVLayerCube.hpp"

static int zero(float x)
{
  if (fabs(x) < .00001) return 1;
  return 0;
}

int main(int argc, char* argv[])
{
  float dx, max;

  max = 2.0;
  
  dx = deltaWithPBC(1., 1., max);
  if ( !zero(dx) ) {
      printf("FAILED:TEST_DELTA: (1.,1.)\n");
      exit(1);
  }

  dx = deltaWithPBC(2., 1., max);
  if ( !zero(-1.0 - dx) ) {
      printf("FAILED:TEST_DELTA: (2.,1.)\n");
      exit(1);
  }

  dx = deltaWithPBC(1., 2., max);
  if ( !zero(1.0 - dx) ) {
      printf("FAILED:TEST_DELTA: (1.,2.)\n");
      exit(1);
  }

  dx = deltaWithPBC(1., 3.0, max);
  if ( !zero(2.0 - fabs(dx)) ) {
      printf("FAILED:TEST_DELTA: (1.,3.)\n");
      exit(1);
  }

  dx = deltaWithPBC(2.4, 0.4, max);
  if ( !zero(2.0 - fabs(dx)) ) {
      printf("FAILED:TEST_DELTA: (2.4,0.4)\n");
      exit(1);
  }

  dx = deltaWithPBC(0., 4., max);
  if ( !zero(0.0 - dx) ) {
      printf("FAILED:TEST_DELTA: (0.,4.)\n");
      exit(1);
  }

  dx = deltaWithPBC(1., 4., max);
  if ( !zero(-1.0 - dx) ) {
      printf("FAILED:TEST_DELTA: (1.,4.)\n");
      exit(1);
  }

  dx = deltaWithPBC(3.4, 1.0, max);
  if ( !zero(1.6 - dx) ) {
      printf("FAILED:TEST_DELTA: (3.4,1.0)\n");
      exit(1);
  }

  max = 2.5;

  dx = deltaWithPBC(3.5, 1.0, max);
  if ( !zero(-2.5 - dx) ) {
      printf("FAILED:TEST_DELTA: (3.5,1.0)\n");
      exit(1);
  }

  dx = deltaWithPBC(3.6, 1.0, max);
  if ( !zero(2.4 - dx) ) {
      printf("FAILED:TEST_DELTA: (3.6,1.0)\n");
      exit(1);
  }

  dx = deltaWithPBC(1.1, 3.7, max);
  if ( !zero(-2.4 - dx) ) {
      printf("FAILED:TEST_DELTA: (1.1,3.7\n");
      exit(1);
  }

  return 0;
}
