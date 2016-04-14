/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <PVsoundRegisterKeywords.hpp>
#include "CochlearLayer.hpp"
#include "StreamReconLayer.h"
#include "inverseCochlearLayer.hpp"
#include "inverseNewCochlearLayer.hpp"
#include "SoundProbe.hpp"

int main(int argc, char * argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false/*do not allow unrecognized arguments*/);
   int status = PV_SUCCESS;
   assert(PV_SUCCESS==0); // |= operator assumes that success is indicated by return value zero.
   status = PVsound::PVsoundRegisterKeywords(&pv_initObj);
   status |= pv_initObj.registerKeyword("CochlearLayer", createCochlearLayer);
   status |= pv_initObj.registerKeyword("StreamReconLayer", createStreamReconLayer);
   status |= pv_initObj.registerKeyword("inverseCochlearLayer", create_inverseCochlearLayer);
   status |= pv_initObj.registerKeyword("inverseNewCochlearLayer", create_inverseNewCochlearLayer);
   status |= buildandrun(&pv_initObj);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
