/*
 * LayerPhaseTestProbe.cpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#include "LayerPhaseTestProbe.hpp"

namespace PV {

LayerPhaseTestProbe::LayerPhaseTestProbe(const char * probename, const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
   PVParams * params = layer->getParent()->parameters();
   equilibriumValue = (pvdata_t) params->value(probename, "equilibriumValue", 0.0f, true);
   equilibriumTime = params->value(probename, "equilibriumTime", 0.0f, true);
}

LayerPhaseTestProbe::LayerPhaseTestProbe(const char * probename, HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
   PVParams * params = layer->getParent()->parameters();
   equilibriumValue = (pvdata_t) params->value(probename, "equilibriumValue", 0.0f, true);
   equilibriumTime = params->value(probename, "equilibriumTime", 0.0f, true);
}


int LayerPhaseTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
   if (timed>=equilibriumTime) {
      double tol = 1e-6;
      assert(fabs(fMin-equilibriumValue) < tol);
      assert(fabs(fMax-equilibriumValue) < tol);
      assert(fabs(avg-equilibriumValue) < tol);
   }

   return status;
}


} /* namespace PV */
