#include <columns/PV_Init.hpp>
#include "MatchingPursuitLayer.hpp"
#include "MatchingPursuitResidual.hpp"
#include "MatchingPursuitRegisterKeywords.hpp"

namespace PVMatchingPursuit {

int MatchingPursuitRegisterKeywords(PV::PV_Init * pv_initObj) {
    int status = PV_SUCCESS;
    assert(PV_SUCCESS==0); // Using the |= operator assumes success is indicated by return value zero.
    status |= pv_initObj->registerKeyword("MatchingPursuitLayer", createMatchingPursuitLayer);
    status |= pv_initObj->registerKeyword("MatchingPursuitResidual", createMatchingPursuitResidual);
    return status;
}

} // end of namespace PVMatchingPursuit
