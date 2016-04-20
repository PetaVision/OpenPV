/*
 * LayerPhaseTestProbe.cpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#include "LayerPhaseTestProbe.hpp"

namespace PV {

LayerPhaseTestProbe::LayerPhaseTestProbe(const char * probeName, HyPerCol * hc)
: StatsProbe()
{
   initLayerPhaseTestProbe_base();
   initLayerPhaseTestProbe(probeName, hc);
}

int LayerPhaseTestProbe::initLayerPhaseTestProbe_base() { return PV_SUCCESS; }

int LayerPhaseTestProbe::initLayerPhaseTestProbe(const char * probeName, HyPerCol * hc) {
   return initStatsProbe(probeName, hc);
}

int LayerPhaseTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_equilibriumValue(ioFlag);
   ioParam_equilibriumTime(ioFlag);
   return status;
}

void LayerPhaseTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufV);
}

void LayerPhaseTestProbe::ioParam_equilibriumValue(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "equilibriumValue", &equilibriumValue, 0.0f, true);
}

void LayerPhaseTestProbe::ioParam_equilibriumTime(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "equilibriumTime", &equilibriumTime, 0.0, true);
}

int LayerPhaseTestProbe::outputState(double timed)
{
   int status = StatsProbe::outputState(timed);
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
   for(int b = 0; b < parent->getNBatch(); b++){
      if (timed>=equilibriumTime) {
         double tol = 1e-6;
         assert(fabs(fMin[b]-equilibriumValue) < tol);
         assert(fabs(fMax[b]-equilibriumValue) < tol);
         assert(fabs(avg[b]-equilibriumValue) < tol);
      }
   }

   return status;
}

BaseObject * createLayerPhaseTestProbe(char const * probeName, HyPerCol * hc) {
   return hc ? new LayerPhaseTestProbe(probeName, hc) : NULL;
}

} /* namespace PV */
