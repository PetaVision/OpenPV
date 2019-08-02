/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "PointProbe.hpp"
#include "layers/HyPerLayer.hpp"
#include <string.h>

namespace PV {

PointProbe::PointProbe() {
   initialize_base();
   // Default constructor for derived classes.  Derived classes should call
   // PointProbe::initialize from their init-method.
}

PointProbe::PointProbe(const char *name, PVParams *params, Communicator const *comm)
      : LayerProbe() {
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
   LayerProbe::initialize(name, params, comm);
}

int PointProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_xLoc(ioFlag);
   ioParam_yLoc(ioFlag);
   ioParam_fLoc(ioFlag);
   ioParam_batchLoc(ioFlag);
   return status;
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
   auto status = LayerProbe::communicateInitInfo(message);
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

void PointProbe::initOutputStreams(const char *filename, Checkpointer *checkpointer) {
   PVLayerLoc const *loc = getTargetLayer()->getLayerLoc();
   int xRank             = xLoc / loc->nx; // integer division
   int yRank             = yLoc / loc->ny; // integer division
   int batchRank         = batchLoc / loc->nbatch; // integer division

   int blockColumnIndex = getMPIBlock()->getStartColumn() + getMPIBlock()->getColumnIndex();
   int blockRowIndex    = getMPIBlock()->getStartRow() + getMPIBlock()->getRowIndex();
   int blockBatchIndex  = getMPIBlock()->getStartBatch() + getMPIBlock()->getBatchIndex();

   if (xRank == blockColumnIndex and yRank == blockRowIndex and batchRank == blockBatchIndex) {
      if (getProbeOutputFilename()) {
         std::string path(getProbeOutputFilename());
         std::ios_base::openmode mode = std::ios_base::out;
         if (!checkpointer->getCheckpointReadDirectory().empty()) {
            mode |= std::ios_base::app;
         }
         if (path[0] != '/') {
            path = checkpointer->makeOutputPathFilename(path);
         }
         auto stream = new FileStream(path.c_str(), mode, checkpointer->doesVerifyWrites());
         mOutputStreams.push_back(stream);
      }
      else {
         auto stream = new PrintStream(PV::getOutputStream());
         mOutputStreams.push_back(stream);
      }
   }
   else {
      mOutputStreams.clear();
   }
}

Response::Status
PointProbe::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = LayerProbe::initializeState(message);
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
   double *valuesBuffer = this->getValuesBuffer();
   auto *globalComm     = mCommunicator->globalCommunicator();

   if (getPointRank() == mCommunicator->globalCommRank()) {
      pvAssert(mPointA);
      valuesBuffer[0] = mPointV ? *mPointV : 0.0f; // mPointV can be null if target layer has no V.
      valuesBuffer[1] = *mPointA;
      // If not in root process, send V and A to root process
      if (mCommunicator->globalCommRank() != 0) {
         MPI_Send(valuesBuffer, 2, MPI_DOUBLE, 0, 0, globalComm);
      }
   }
   else {
      pvAssert(mPointA == nullptr and mPointV == nullptr);
   }

   // Root process receives from local rank of the target point.
   if (mCommunicator->globalCommRank() == 0 and getPointRank() != 0) {
      MPI_Recv(valuesBuffer, 2, MPI_DOUBLE, getPointRank(), 0, globalComm, MPI_STATUS_IGNORE);
   }
}

void PointProbe::writeState(double timevalue) {
   if (!mOutputStreams.empty()) {
      double const V = getValuesBuffer()[0];
      double const A = getValuesBuffer()[1];
      output(0).printf("%s t=%.1f V=%6.5f a=%.5f", getMessage(), timevalue, V, A);
      output(0) << std::endl;
   }
}

} // namespace PV
