/*
 * DiscreteUniformRandomV.cpp
 *
 *  Created on: Sept 28, 2022
 *      Author: peteschultz
 */

#include "DiscreteUniformRandomV.hpp"
#include "columns/Random.hpp"
#include "utils/PVLog.hpp"
#include <cmath>

namespace PV {

DiscreteUniformRandomV::DiscreteUniformRandomV() { initialize_base(); }

DiscreteUniformRandomV::DiscreteUniformRandomV(char const *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

DiscreteUniformRandomV::~DiscreteUniformRandomV() {}

int DiscreteUniformRandomV::initialize_base() { return PV_SUCCESS; }

void DiscreteUniformRandomV::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseInitV::initialize(name, params, comm);
}

int DiscreteUniformRandomV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_minV(ioFlag);
   ioParam_maxV(ioFlag);
   ioParam_numValues(ioFlag);
   FatalIf(
         mMaxV < mMinV,
         "%s has maxV=%f less than minV=%f.\n",
         getDescription().c_str(),
         (double)mMaxV,
         (double)mMinV);
   return status;
}

void DiscreteUniformRandomV::ioParam_minV(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "minV", &mMinV, mMinV);
}

void DiscreteUniformRandomV::ioParam_maxV(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "minV"));
   parameters()->ioParamValue(ioFlag, name, "maxV", &mMaxV, mMinV + 1.0f);
}

void DiscreteUniformRandomV::ioParam_numValues(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "numValues", &mNumValues, mNumValues);
   FatalIf(
         mNumValues < 2,
         "%s parameter \"numValues\" is %d, but it must be at least 2.\n",
         getDescription().c_str(),
         mNumValues);
}

void DiscreteUniformRandomV::calcV(float *V, PVLayerLoc const *loc) {
   PVLayerLoc flatLoc;
   memcpy(&flatLoc, loc, sizeof(PVLayerLoc));
   flatLoc.nf = 1;
   Random randState{&flatLoc, false /*not extended*/};
   int const nxny = flatLoc.nx * flatLoc.ny;
   double minV = static_cast<double>(mMinV);
   double maxV = static_cast<double>(mMaxV);
   double numValues = static_cast<double>(mNumValues);
   double dV = (maxV - minV) / (numValues - 1.0);
   double p = numValues / (1.0 + static_cast<double>(CL_RANDOM_MAX));
   for (int b = 0; b < loc->nbatch; b++) {
      float *VBatch = V + b * loc->nx * loc->ny * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int xy = 0; xy < nxny; xy++) {
         for (int f = 0; f < loc->nf; f++) {
            int index        = kIndex(xy, 0, f, nxny, 1, loc->nf);
            double randomInt = std::floor(p * static_cast<double>(randState.randomUInt(xy)));
            double value     = minV + dV * randomInt;
            VBatch[index]    = value;
         }
      }
   }
}

} // end namespace PV
