/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>
#include "CustomGroupHandler.hpp"

int main(int argc, char * argv[]) {

   int status;
   ParamGroupHandler * customGroupHandler = new CustomGroupHandler();
   status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1/*numGroupHandlers*/);
   delete customGroupHandler;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
