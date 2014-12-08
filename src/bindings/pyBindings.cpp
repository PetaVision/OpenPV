#include "../columns/buildandrun.hpp"

extern "C" {
   HyPerCol * pvBuild(int argc, char* argv[]){ return build(argc, argv);}
   int pvRun(HyPerCol* hc){return hc->run();}
}
