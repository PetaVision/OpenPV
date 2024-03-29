/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "PointProbe.hpp"
#include "arch/mpi/mpi.h"
#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "components/ActivityComponent.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/InternalStateBuffer.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_common.h"
#include "io/FileStreamBuilder.hpp"
#include "io/PrintStream.hpp"
#include "layers/HyPerLayer.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include <cassert>
#include <cstdlib>
#include <ostream>
#include <string>

namespace PV {

PointProbe::PointProbe() {
   initialize_base();
   // Default constructor for derived classes.  Derived classes should call
   // PointProbe::initialize from their init-method.
}

PointProbe::PointProbe(const char *name, PVParams *params, Communicator const *comm)
      : LegacyLayerProbe() {
   initialize_base();
   initialize(name, params, comm);
}

PointProbe::~PointProbe() {}

int PointProbe::initialize_base() {
   xLoc     = 0;
   yLoc     = 0;
   fLoc     = 0;
   batchLoc = 0;
   return PV_SUCCESS;
}

void PointProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   LegacyLayerProbe::initialize(name, params, comm);
}

int PointProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LegacyLayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_xLoc(ioFlag);
   ioParam_yLoc(ioFlag);
   ioParam_fLoc(ioFlag);
   ioParam_batchLoc(ioFlag);
   return status;
}

void PointProbe::ioParam_statsFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "statsFlag");
   }
}

void PointProbe::ioParam_xLoc(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "xLoc", &xLoc);
}

void PointProbe::ioParam_yLoc(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "yLoc", &yLoc);
}

void PointProbe::ioParam_fLoc(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "fLoc", &fLoc);
}

void PointProbe::ioParam_batchLoc(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "batchLoc", &batchLoc);
}

void PointProbe::initNumValues() { setNumValues(2); }

Response::Status
PointProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = LegacyLayerProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   assert(getTargetLayer());
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   bool isRoot           = mCommunicator->commRank() == 0;
   bool failed           = false;
   if ((xLoc < 0 || xLoc > loc->nxGlobal) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: xLoc coordinate %d is out "
            "of bounds (layer has %d neurons in "
            "the x-direction.\n",
            getTargetLayer()->getName(),
            xLoc,
            loc->nxGlobal);
      failed = true;
   }
   if ((yLoc < 0 || yLoc > loc->nyGlobal) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: yLoc coordinate %d is out "
            "of bounds (layer has %d neurons in "
            "the y-direction.\n",
            getTargetLayer()->getName(),
            yLoc,
            loc->nyGlobal);
      failed = true;
   }
   if ((fLoc < 0 || fLoc > loc->nf) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: fLoc coordinate %d is out "
            "of bounds (layer has %d features.\n",
            getTargetLayer()->getName(),
            fLoc,
            loc->nf);
      failed = true;
   }
   if ((batchLoc < 0 || batchLoc > loc->nbatchGlobal) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: batchLoc coordinate %d is "
            "out of bounds (layer has %d "
            "batch elements.\n",
            getTargetLayer()->getName(),
            batchLoc,
            loc->nbatch);
      failed = true;
   }
   if (failed)
      exit(EXIT_FAILURE);
   return Response::SUCCESS;
}

void PointProbe::initOutputStreams(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   PVLayerLoc const *loc = getTargetLayer()->getLayerLoc();
   int xRank             = xLoc / loc->nx; // integer division
   int yRank             = yLoc / loc->ny; // integer division
   int batchRank         = batchLoc / loc->nbatch; // integer division

   auto ioMPIBlock      = getCommunicator()->getIOMPIBlock();
   int blockColumnIndex = ioMPIBlock->getStartColumn() + ioMPIBlock->getColumnIndex();
   int blockRowIndex    = ioMPIBlock->getStartRow() + ioMPIBlock->getRowIndex();
   int blockBatchIndex  = ioMPIBlock->getStartBatch() + ioMPIBlock->getBatchIndex();

   if (xRank == blockColumnIndex and yRank == blockRowIndex and batchRank == blockBatchIndex) {
      if (getProbeOutputFilename()) {
         auto *checkpointer              = message->mDataRegistry;
         char const *probeOutputFilename = getProbeOutputFilename();
         std::string path(probeOutputFilename);
         bool createFlag = checkpointer->getCheckpointReadDirectory().empty();
         auto stream     = FileStreamBuilder(
                             getCommunicator()->getOutputFileManager(),
                             getProbeOutputFilename(),
                             true /*text*/,
                             false /*not read-only*/,
                             createFlag,
                             checkpointer->doesVerifyWrites())
                             .get();
         auto checkpointEntry =
               std::make_shared<CheckpointEntryFilePosition>(getProbeOutputFilename(), stream);
         bool registerSucceeded = checkpointer->registerCheckpointEntry(
               checkpointEntry, false /*not constant for entire run*/);
         FatalIf(
               !registerSucceeded,
               "%s failed to register %s for checkpointing.\n",
               getDescription_c(),
               checkpointEntry->getName().c_str());
         mOutputStreams.push_back(stream);
      }
      else {
         auto stream = std::make_shared<PrintStream>(PV::getOutputStream());
         mOutputStreams.push_back(stream);
      }
   }
   else {
      mOutputStreams.clear();
   }
}

Response::Status
PointProbe::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = LegacyLayerProbe::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   // We need to calculate which mpi process contains the target point, and send
   // that info to the root process Each process calculates local index
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();

   // Compute rank for which target point is in the restricted region.
   int xRank      = xLoc / loc->nx;
   int yRank      = yLoc / loc->ny;
   int bRank      = batchLoc / loc->nbatch;
   int numRows    = mCommunicator->numCommRows();
   int numColumns = mCommunicator->numCommColumns();
   int batchWidth = mCommunicator->numCommBatches();
   mPointRank     = rankFromRowColumnBatch(yRank, xRank, bRank, numRows, numColumns, batchWidth);

   mPointV = nullptr;
   mPointA = nullptr;

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

      const int k = kIndex(xLocLocal, yLocLocal, fLoc, nx, ny, nf);

      auto *activityComponent = getTargetLayer()->getComponentByType<ActivityComponent>();
      auto *internalState     = activityComponent->getComponentByType<InternalStateBuffer>();
      if (internalState != nullptr) {
         int offsetV = k + nbatchLocal * internalState->getBufferSize();
         mPointV     = &internalState->getBufferData()[offsetV];
      }

      auto *publisherComponent = getTargetLayer()->getComponentByType<BasePublisherComponent>();
      if (publisherComponent != nullptr) {
         float const *activity = publisherComponent->getLayerData();
         const int kex         = kIndexExtended(
               k, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         int offsetA = kex + nbatchLocal * getTargetLayer()->getNumExtended();
         mPointA     = &activity[kex];
      }
   }

   return Response::SUCCESS;
}

/**
 * NOTES:
 *     - Only the activity buffer covers the extended frame - this is the frame
 * that
 * includes boundaries.
 *     - The membrane potential V covers the "real" or "restricted" frame.
 *     - In MPI runs, xLoc and yLoc refer to global coordinates.
 *     writeState is only called by the processor with (xLoc,yLoc) in its
 *     non-extended region.
 */
Response::Status PointProbe::outputState(double simTime, double deltaTime) {
   getValues(simTime);
   writeState(simTime);
   return Response::SUCCESS;
}

void PointProbe::calcValues(double timevalue) {
   assert(this->getNumValues() == 2);
   auto &valuesVector = getProbeValues();
   auto globalComm    = mCommunicator->globalCommunicator();

   if (getPointRank() == mCommunicator->globalCommRank()) {
      pvAssert(mPointA);
      valuesVector[0] = mPointV ? *mPointV : 0.0f; // mPointV can be null if target layer has no V.
      valuesVector[1] = *mPointA;
      // If not in root process, send V and A to root process
      if (mCommunicator->globalCommRank() != 0) {
         MPI_Send(valuesVector.data(), 2, MPI_DOUBLE, 0, 0, globalComm);
      }
   }
   else {
      pvAssert(mPointA == nullptr and mPointV == nullptr);
   }

   // Root process receives from local rank of the target point.
   if (mCommunicator->globalCommRank() == 0 and getPointRank() != 0) {
      MPI_Recv(
            valuesVector.data(), 2, MPI_DOUBLE, getPointRank(), 0, globalComm, MPI_STATUS_IGNORE);
   }
}

void PointProbe::writeState(double timevalue) {
   if (!mOutputStreams.empty()) {
      double const V = getProbeValues()[0];
      double const A = getProbeValues()[1];
      output(0).printf("%s t=%.1f V=%6.5f a=%.5f", getMessage(), timevalue, V, A);
      output(0) << std::endl;
   }
}

Response::Status PointProbe::outputStateStats(double simTime, double deltaTime) {
   Fatal() << "PointProbe::outputStateStats() should never be called.\n";
   return Response::NO_ACTION; // to suppress compiler warnings
}

} // namespace PV
