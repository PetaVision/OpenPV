/*
 * UniformRandomV.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "UniformRandomV.hpp"
#include "columns/Random.hpp"
#include "utils/PVLog.hpp"

namespace PV {

UniformRandomV::UniformRandomV() { initialize_base(); }

UniformRandomV::UniformRandomV(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize_base();
   initialize(params, defaults, comm);
}

UniformRandomV::~UniformRandomV() {}

int UniformRandomV::initialize_base() { return PV_SUCCESS; }

void UniformRandomV::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseInitV::initialize(params, defaults, comm);
}

int UniformRandomV::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = BaseInitV::ioParamsFillGroup(ioSwitch);
   ioParam_minV(ioSwitch);
   ioParam_maxV(ioSwitch);
   return status;
}

void UniformRandomV::ioParam_minV(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "minV", &mMinV);
}

void UniformRandomV::ioParam_maxV(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("minV"));
   if (mParamsIO->isPresent("maxV")) {
      mParamsIO->ioParam(ioSwitch, "maxV", &mMaxV);
   }
   else {
      switch (ioSwitch) {
         case ParamsIOSwitch::Read:
            mMaxV = mMinV + 1.0f;
            WarnLog().printf(
                  "Using inferred value %f for parameter %s in group \"%s\"\n",
                  (double)mMaxV, "maxV", getName());
            break;
         case ParamsIOSwitch::Write:
            mParamsIO->ioParam(ioSwitch, "maxV", &mMaxV);
            break;
         default:
            Fatal().printf("Unrecognized ParamsIOFlag %d\n", ioSwitch);
            break;
      }
   }
}

void UniformRandomV::calcV(float *V, PVLayerLoc const *loc) {
   PVLayerLoc flatLoc;
   memcpy(&flatLoc, loc, sizeof(PVLayerLoc));
   flatLoc.nf = 1;
   Random randState{&flatLoc, false /*not extended*/};
   const int nxny = flatLoc.nx * flatLoc.ny;
   for (int b = 0; b < loc->nbatch; b++) {
      float *VBatch = V + b * loc->nx * loc->ny * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int xy = 0; xy < nxny; xy++) {
         for (int f = 0; f < loc->nf; f++) {
            int index     = kIndex(xy, 0, f, nxny, 1, loc->nf);
            VBatch[index] = randState.uniformRandom(xy, mMinV, mMaxV);
         }
      }
   }
}

} // end namespace PV
