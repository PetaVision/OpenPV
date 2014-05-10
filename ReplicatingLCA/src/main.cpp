/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
