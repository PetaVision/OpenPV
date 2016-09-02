/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>

int main(int argc, char * argv[]) {
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   int status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
