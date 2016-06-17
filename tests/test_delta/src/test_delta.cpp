#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "utils/PVLog.hpp"
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
      pvError().printf("FAILED:TEST_DELTA: (1.,1.)\n");
  }

  dx = deltaWithPBC(2., 1., max);
  if ( !zero(-1.0 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (2.,1.)\n");
  }

  dx = deltaWithPBC(1., 2., max);
  if ( !zero(1.0 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (1.,2.)\n");
  }

  dx = deltaWithPBC(1., 3.0, max);
  if ( !zero(2.0 - fabs(dx)) ) {
      pvError().printf("FAILED:TEST_DELTA: (1.,3.)\n");
  }

  dx = deltaWithPBC(2.4, 0.4, max);
  if ( !zero(2.0 - fabs(dx)) ) {
      pvError().printf("FAILED:TEST_DELTA: (2.4,0.4)\n");
  }

  dx = deltaWithPBC(0., 4., max);
  if ( !zero(0.0 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (0.,4.)\n");
  }

  dx = deltaWithPBC(1., 4., max);
  if ( !zero(-1.0 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (1.,4.)\n");
  }

  dx = deltaWithPBC(3.4, 1.0, max);
  if ( !zero(1.6 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (3.4,1.0)\n");
  }

  max = 2.5;

  dx = deltaWithPBC(3.5, 1.0, max);
  if ( !zero(-2.5 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (3.5,1.0)\n");
  }

  dx = deltaWithPBC(3.6, 1.0, max);
  if ( !zero(2.4 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (3.6,1.0)\n");
  }

  dx = deltaWithPBC(1.1, 3.7, max);
  if ( !zero(-2.4 - dx) ) {
      pvError().printf("FAILED:TEST_DELTA: (1.1,3.7\n");
  }

  return 0;
}
