/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include "BIDSGroupHandler.hpp"

int main(int argc, char * argv[]) {
   ParamGroupHandler * bidsGroupHandler = new PVBIDS::BIDSGroupHandler();
   int status = buildandrun(argc, argv, NULL, NULL, &bidsGroupHandler, 1);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
