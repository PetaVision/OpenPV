/*
 * UpdateFromCloneTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include "UpdateFromCloneTestGroupHandler.hpp"

int main(int argc, char * argv[]) {

   int status;
   ParamGroupHandler * customGroupHandler = new UpdateFromCloneTestGroupHandler;
   status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
