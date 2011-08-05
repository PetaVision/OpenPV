/*
 * test_kernel.cpp
 *
 ** created by garkenyon: August 4, 2011
 */

#include "../src/columns/buildandrun.hpp"

int test_kernel(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
    return buildandrun(argc, argv, &test_kernel)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int test_kernel(HyPerCol * hc, int argc, char * argv[]){
   int status = 0;
   return status;
}
