/*
 * RescaleLayerTest.cpp
 *
 */
#include "columns/buildandrun.hpp"
#include <cstdlib>

int main(int argc, char *argv[]) {
   int status = buildandrun(argc, argv);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
