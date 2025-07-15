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

UniformRandomV::UniformRandomV(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize_base();
   initialize(paramsIO, comm);
}

UniformRandomV::~UniformRandomV() {}

int UniformRandomV::initialize_base() { return PV_SUCCESS; }

void UniformRandomV::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   BaseInitV::initialize(paramsIO, comm);
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
   mParamsIO->ioParam(ioSwitch, "maxV", &mMaxV);
   FatalIf(
         mMaxV < mMinV,
         "%s \"%s\" with UniformRandomV has maxV = %f < minV = %f\n",
         mParamsIO->getKeyword(), mParamsIO->getName(), (double)mMaxV, (double)mMinV);
}

void UniformRandomV::calcV(float *V, PVLayerLoc const *loc) {
   pvAssert(mMaxV >= mMinV); // checked when reading params
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
