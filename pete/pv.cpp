/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"

int main(int argc, char * argv[]) {
    return buildandrun(argc, argv)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
