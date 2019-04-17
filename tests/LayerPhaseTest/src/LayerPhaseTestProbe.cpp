/*
 * LayerPhaseTestProbe.cpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#include "LayerPhaseTestProbe.hpp"

namespace PV {

LayerPhaseTestProbe::LayerPhaseTestProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm)
      : StatsProbe() {
   initialize_base();
   initialize(name, params, comm);
}

int LayerPhaseTestProbe::initialize_base() { return PV_SUCCESS; }

void LayerPhaseTestProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
}

int LayerPhaseTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_equilibriumValue(ioFlag);
   ioParam_equilibriumTime(ioFlag);
   return status;
}

void LayerPhaseTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) { requireType(BufV); }

void LayerPhaseTestProbe::ioParam_equilibriumValue(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "equilibriumValue", &equilibriumValue, 0.0f, true);
}

void LayerPhaseTestProbe::ioParam_equilibriumTime(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "equilibriumTime", &equilibriumTime, 0.0, true);
}

Response::Status LayerPhaseTestProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (status != Response::SUCCESS) {
      return status;
   }
   const int rcvProc = 0;
   if (mCommunicator->commRank() != rcvProc) {
      return status;
   }
   for (int b = 0; b < mLocalBatchWidth; b++) {
      if (simTime >= equilibriumTime) {
         float const tol = 1e-6f;
         // TODO: std::fabs is preferred to fabsf. But we implicitly include
         // math.h because the header includes HyPerLayer.hpp, which eventually
         // includes cl_random.h. It seems iffy to include both math.h and cmath.
         // We should convert cl_random.{c,h} to .cpp and .hpp.
         FatalIf(fabsf(fMin[b] - equilibriumValue) >= tol, "Test failed.\n");
         FatalIf(fabsf(fMax[b] - equilibriumValue) >= tol, "Test failed.\n");
         FatalIf(fabsf(avg[b] - equilibriumValue) >= tol, "Test failed.\n");
      }
   }

   return status;
}

} /* namespace PV */
