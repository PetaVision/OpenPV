#include <columns/buildandrun.hpp>
#include "AudioGroupHandler.hpp"

int main(int argc, char * argv[])
{
   ParamGroupHandler * customGroupHandlers[1];
   customGroupHandlers[0] = new AudioGroupHandler;
   int status = buildandrun(argc, argv, NULL, NULL, customGroupHandlers, 1);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
