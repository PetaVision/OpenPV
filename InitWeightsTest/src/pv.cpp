/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include "InitWeightsTestParamGroupHandler.hpp"

int main(int argc, char * argv[]) {
   ParamGroupHandler * customGroupHandler = new InitWeightsTestParamGroupHandler();
   assert(customGroupHandler != NULL);
   int status = buildandrun(argc, argv, NULL, NULL, &customGroupHandler, 1);
   delete customGroupHandler;
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
