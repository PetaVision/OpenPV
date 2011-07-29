/*
 * pv.cpp
 *
 */

#include "../PetaVision/src/columns/buildandrun.hpp"
#include "ChannelProbe.hpp"

#define MAIN_USES_ADDCUSTOM

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

	int status;
#ifdef MAIN_USES_ADDCUSTOM
	status = buildandrun(argc, argv, &addcustom);
#else
    status = buildandrun(argc, argv);
#endif // MAIN_USES_ADDCUSTOM
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]) {
    HyPerLayer * retina = hc->getLayer(1);

    assert(!strcmp(retina->getName(), "Retina"));

    ChannelProbe * retina_exc_channelprobe = new ChannelProbe("Retina_Exc_ChannelProbe.txt", hc, CHANNEL_EXC);
    retina->insertProbe(retina_exc_channelprobe);

    return PV_SUCCESS;
}
#endif // MAIN_USES_ADDCUSTOM
