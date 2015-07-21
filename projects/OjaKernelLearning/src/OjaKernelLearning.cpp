//============================================================================
// OjaKernelLearning
// Author: peteschultz
// A PetaVision project for learning dictionaries using OjaKernelConn and LCALIFLateralKernelConn
//============================================================================

#include <columns/buildandrun.hpp>

void * customgroups(const char * keyword, const char * groupname, HyPerCol * hc);

int main(int argc, char * argv[]) {

    int status;
    status = buildandrun(argc, argv, NULL, NULL, &customgroups);
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void * customgroups(const char * keyword, const char * groupname, HyPerCol * hc) {
   return NULL;
}
