/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <columns/PV_Init.hpp>
#include "TestAllZerosProbe.hpp"

int main(int argc, char * argv[]) {
   PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   pv_initObj.registerKeyword("TestAllZerosProbe", Factory::standardCreate<TestAllZerosProbe>);
   int status = buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
