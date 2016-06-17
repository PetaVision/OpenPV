#include "layers/PVLayerCube.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/PVLog.hpp"

static int zero(float x)
{
  if (fabs(x) < .00001) return 1;
  return 0;
}

int main(int argc, char* argv[])
{
  float s;

  s = sign(3.3);
  if ( !zero(1.0 - s) ) {
      pvError().printf("FAILED:TEST_SIGN: (3.3)\n");
  }

  s = sign(.001);
  if ( !zero(1.0 - s) ) {
      pvError().printf("FAILED:TEST_SIGN: (.001)\n");
  }

  s = sign(-.001);
  if ( !zero(-1.0 - s) ) {
      pvError().printf("FAILED:TEST_SIGN: (-.001)\n");
  }

  s = sign(0.0);
  if ( !zero(1.0 - s) ) {
      pvError().printf("FAILED:TEST_SIGN: (0.0)\n");
  }

  return 0;
}
