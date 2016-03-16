/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "ArborTestProbe.hpp"
#include "ArborTestForOnesProbe.hpp"
#include "ArborSystemTestGroupHandler.hpp"

int main(int argc, char * argv[]) {
   ParamGroupHandler * customGroupHandler = new ArborSystemTestGroupHandler();
   int status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
