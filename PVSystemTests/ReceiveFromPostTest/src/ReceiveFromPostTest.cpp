/*
 * ReceiveFromPostTest
 *
 *
 */


#include <columns/buildandrun.hpp>
#include "CustomGroupHandler.hpp"

int main(int argc, char * argv[]) {

   ParamGroupHandler * customGroupHandler = new CustomGroupHandler;
   int status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1);
   delete customGroupHandler;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
