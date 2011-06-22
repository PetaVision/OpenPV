/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "ChannelProbe.hpp"
#include "VProbe.hpp"

#undef MAIN_USES_BUILD
#define MAIN_USES_BUILDANDRUN

int addcustom(HyPerCol * hc, int argc, char * argv[]);
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

int addcustom(HyPerCol * hc, int argc, char * argv[]) {
    HyPerLayer * anaretina = hc->getLayer(2);
    HyPerLayer * layera = hc->getLayer(3);
    HyPerLayer * pooling_analayera = hc->getLayer(4);
    HyPerLayer * arrow_catalayerb = hc->getLayer(8);

    assert(!strcmp(anaretina->getName(), "AnaRetina"));
    assert(!strcmp(layera->getName(), "Layer A"));
    assert(!strcmp(pooling_analayera->getName(), "Pooling AnaLayer A"));
    assert(!strcmp(arrow_catalayerb->getName(), "Arrow CataLayer B"));

    VProbe * anaretina_vprobe = new VProbe("AnaRetina_VProbe.txt", hc);
    anaretina->insertProbe(anaretina_vprobe);

    VProbe * pooling_analayera_vprobe = new VProbe("Pooling_AnaLayer_A_VProbe.txt", hc);
    pooling_analayera->insertProbe(pooling_analayera_vprobe);

    VProbe * arrow_catalayerb_vprobe = new VProbe("Arrow_CataLayer_B_VProbe.txt", hc);
    arrow_catalayerb->insertProbe(arrow_catalayerb_vprobe);

    ChannelProbe * layera_channelexc_probe = new ChannelProbe("Layer_A_Exc_ChannelProbe.txt", hc, CHANNEL_EXC);
    layera->insertProbe(layera_channelexc_probe);
    ChannelProbe * layera_channelinh_probe = new ChannelProbe("Layer_A_Inh_ChannelProbe.txt", hc, CHANNEL_INH);
    layera->insertProbe(layera_channelinh_probe);

    return PV_SUCCESS;
}
