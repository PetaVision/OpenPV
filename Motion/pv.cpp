/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
//#include "ChannelProbe.hpp"
//#include "VProbe.hpp"

#undef MAIN_USES_BUILD
#define MAIN_USES_BUILDANDRUN

//int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
// To use addcustom, undef MAIN_USES_BUILDANDRUN and def MAIN_USES_BUILD

int main(int argc, char * argv[]) {
#ifdef MAIN_USES_BUILDANDRUN
    return buildandrun(argc, argv)==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
#endif // MAIN_USES_BUILDANDRUN

#ifdef MAIN_USES_BUILD
    HyPerCol * hc = build(argc, argv);
    if( hc == NULL ) return EXIT_FAILURE;
    int status;
    status = addcustom(hc, argc, argv);
    if( status != PV_SUCCESS ) return status;
    if( hc->numberOfTimeSteps() > 0 ) {
        status = hc->run();
        if( status != PV_SUCCESS ) {
            fprintf(stderr, "HyPerCol::run() returned with error code %d\n", status);
        }
    }
    delete hc; /* HyPerCol's destructor takes care of deleting layers and connections */
    return status;
#endif // MAIN_USES_BUILD
}

