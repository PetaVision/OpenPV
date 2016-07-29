#include <stddef.h>
#include "../columns/buildandrun.hpp"

extern "C" {
   HyPerCol * pvBuild(int argc, char* argv[]){ 
      PV_Init * initObj = new PV_Init(&argc, &argv, false/*allowUnrecognizedArguments*/);
      return build(initObj);
   }
   int pvRun(HyPerCol* hc){return hc->run();}
}
