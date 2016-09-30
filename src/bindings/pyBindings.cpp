#include "../columns/buildandrun.hpp"
#include <stddef.h>

extern "C" {
HyPerCol *pvBuild(int argc, char *argv[]) {
   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   return build(initObj);
}
int pvRun(HyPerCol *hc) { return hc->run(); }
}
