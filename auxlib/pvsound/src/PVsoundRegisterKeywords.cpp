#include "PVsoundRegisterKeywords.hpp"
#include "NewCochlear.h"
#include "SoundStream.hpp"

namespace PVsound {

int PVsoundRegisterKeywords(PV::PV_Init * pv_initObj) {
    int status = PV_SUCCESS;
    assert(PV_SUCCESS==0); // |= operator assumes success is indicated by return value 0.
    status |= pv_initObj->registerKeyword("NewCochlearLayer", createNewCochlearLayer);
    status |= pv_initObj->registerKeyword("SoundStream", createSoundStream);
    return status;
}

} // end of namespace PVsound
