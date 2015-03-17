/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <PVsoundGroupHandler.hpp>
#include "soundAnalysisGroupHandler.hpp"

int main(int argc, char * argv[]) {
   ParamGroupHandler * customGroupHandlers[2];
   customGroupHandlers[0] = new PVsound::PVsoundGroupHandler;
   customGroupHandlers[1] = new soundAnalysisGroupHandler;
   int status = buildandrun(argc, argv, NULL, NULL, customGroupHandlers, 2);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
