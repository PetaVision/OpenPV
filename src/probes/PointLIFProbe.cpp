/*
 * PointLIFProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "PointLIFProbe.hpp"
#include "components/LIFActivityComponent.hpp"
#include "layers/HyPerLayer.hpp"
#include <assert.h>
#include <string.h>

#define NUMBER_OF_VALUES 6 // G_E, G_I, G_IB, V, Vth, A
#define CONDUCTANCE_PRINT_FORMAT "%6.3f"

namespace PV {

PointLIFProbe::PointLIFProbe() : PointProbe() {
   // Derived classes of PointLIFProbe should use this PointLIFProbe constructor,
   // and call PointLIFProbe::initialize during their initialization.
}

PointLIFProbe::PointLIFProbe(const char *name, PVParams *params, Communicator const *comm)
      : PointProbe() {
   initialize(name, params, comm);
}

void PointLIFProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   PointProbe::initialize(name, params, comm);
   writeTime = 0.0;
}

int PointLIFProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PointProbe::ioParamsFillGroup(ioFlag);
   ioParam_writeStep(ioFlag);
   return status;
}

void PointLIFProbe::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   // If writeStep is not set in params, we initialize it to zero here; in the
   // CommunicateInitInfo state, we set it to the parent's DeltaTime.
   // If writing a derived class that overrides ioParam_writeStep, check if the
   // setDefaultWriteStep method also needs to be overridden.
   writeStep         = 0.0;
   bool warnIfAbsent = false;
   parameters()->ioParamValue(ioFlag, name, "writeStep", &writeStep, writeStep, warnIfAbsent);
}

void PointLIFProbe::initNumValues() { setNumValues(NUMBER_OF_VALUES); }

Response::Status
PointLIFProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = PointProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (!parameters()->present(getName(), "writeStep")) {
      setDefaultWriteStep(message);
   }
   return Response::SUCCESS;
}

void PointLIFProbe::setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   writeStep = message->mDeltaTime;
   // Call ioParamValue to generate the warnIfAbsent warning.
   parameters()->ioParamValue(PARAMS_IO_READ, name, "writeStep", &writeStep, writeStep, true);
}

Response::Status
PointLIFProbe::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = PointProbe::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (getPointRank() == mCommunicator->globalCommRank()) {
      PVLayerLoc const *loc = getTargetLayer()->getLayerLoc();
      int const nx          = loc->nx;
      int const ny          = loc->ny;
      int const nf          = loc->nf;
      int const nbatch      = loc->nbatch;
      // Calculate local coords from global
      int const xLocLocal   = xLoc - loc->kx0;
      int const yLocLocal   = yLoc - loc->ky0;
      int const nbatchLocal = batchLoc - loc->kb0;

      // For the correct rank, target point should be in the local restricted region.
      pvAssert(xLocLocal >= 0 and xLocLocal < nx and yLocLocal >= 0 and yLocLocal < ny);
      pvAssert(nbatchLocal >= 0 and nbatchLocal < nbatch);

      int const k                = kIndex(xLocLocal, yLocLocal, fLoc, nx, ny, nf);
      int const numNeuronsGlobal = loc->nxGlobal * loc->nyGlobal * loc->nf;
      int const offsetRestricted = k + nbatchLocal * numNeuronsGlobal;

      auto *activityComponent = getTargetLayer()->getComponentByType<ActivityComponent>();
      auto buffers = activityComponent->getTable()->findObjects<ComponentBuffer>(getName());
      for (auto &buf : buffers) {
         std::string label = buf->getBufferLabel();
         if (label == "") {
            continue;
         }
         else if (label == "G_E") {
            mPointG_E = &buf->getBufferData()[offsetRestricted];
            continue;
         }
         else if (label == "G_I") {
            mPointG_I = &buf->getBufferData()[offsetRestricted];
            continue;
         }
         else if (label == "G_IB") {
            mPointG_IB = &buf->getBufferData()[offsetRestricted];
            continue;
         }
         else if (label == "Vth") {
            mPointVth = &buf->getBufferData()[offsetRestricted];
            continue;
         }
         pvAssert(getPointV()); // LIFLayers must have a V component. mPointV set by PointProbe.
      }
   }

   return Response::SUCCESS;
}

void PointLIFProbe::calcValues(double timevalue) {
   assert(this->getNumValues() == NUMBER_OF_VALUES);
   double *valuesBuffer = this->getValuesBuffer();
   auto *globalComm     = mCommunicator->globalCommunicator();

   // if in bounds
   if (getPointRank() == mCommunicator->globalCommRank()) {
      pvAssert(getPointA() and getPointV());
      pvAssert(getPointG_E() and getPointG_I() and getPointG_IB() and getPointVth());
      valuesBuffer[0] = *mPointG_E;
      valuesBuffer[1] = *mPointG_I;
      valuesBuffer[2] = *mPointG_IB;
      valuesBuffer[3] = *getPointV();
      valuesBuffer[4] = *mPointVth;
      valuesBuffer[5] = *getPointA();
      // If not in root process, send values to root process
      if (mCommunicator->globalCommRank() != 0) {
         MPI_Send(valuesBuffer, NUMBER_OF_VALUES, MPI_DOUBLE, 0, 0, globalComm);
      }
   }

   // Root process receives from local rank of the target point.
   if (mCommunicator->globalCommRank() == 0 and getPointRank() != 0) {
      auto *globalComm = mCommunicator->globalCommunicator();
      MPI_Recv(
            valuesBuffer,
            NUMBER_OF_VALUES,
            MPI_DOUBLE,
            getPointRank(),
            0,
            globalComm,
            MPI_STATUS_IGNORE);
   }
}

/**
 * @time
 * @l
 * @k
 * @kex
 * NOTES:
 *     - Only the activity buffer covers the extended frame - this is the frame
 * that
 * includes boundaries.
 *     - The other dynamic variables (G_E, G_I, V, Vth) cover the "real" or
 * "restricted"
 *     frame.
 */
void PointLIFProbe::writeState(double timevalue) {
   if (!mOutputStreams.empty() and timevalue >= writeTime) {
      writeTime += writeStep;
      PVLayerLoc const *loc = getTargetLayer()->getLayerLoc();
      const int k           = kIndex(xLoc, yLoc, fLoc, loc->nxGlobal, loc->nyGlobal, loc->nf);
      double *valuesBuffer  = getValuesBuffer();
      output(0).printf(
            "%s t=%.1f index %d batchelement %d "
            "G_E=" CONDUCTANCE_PRINT_FORMAT " G_I=" CONDUCTANCE_PRINT_FORMAT
            " G_IB=" CONDUCTANCE_PRINT_FORMAT " V=" CONDUCTANCE_PRINT_FORMAT
            " Vth=" CONDUCTANCE_PRINT_FORMAT " a=%.1f",
            getMessage(),
            timevalue,
            k,
            batchLoc,
            valuesBuffer[0],
            valuesBuffer[1],
            valuesBuffer[2],
            valuesBuffer[3],
            valuesBuffer[4],
            valuesBuffer[5]);
      output(0) << std::endl;
   }
}

} // namespace PV
