/*
 * PointProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: Craig Rasmussen
 */

#include "PointProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include <string.h>

namespace PV {

PointProbe::PointProbe() {
   initialize_base();
   // Default constructor for derived classes.  Derived classes should call
   // PointProbe::initialize from their init-method.
}

PointProbe::PointProbe(const char *name, HyPerCol *hc) : LayerProbe() {
   initialize_base();
   initialize(name, hc);
}

PointProbe::~PointProbe() {}

int PointProbe::initialize_base() {
   xLoc     = 0;
   yLoc     = 0;
   fLoc     = 0;
   batchLoc = 0;
   return PV_SUCCESS;
}

int PointProbe::initialize(const char *name, HyPerCol *hc) {
   int status = LayerProbe::initialize(name, hc);
   return status;
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
   parent->parameters()->ioParamValueRequired(ioFlag, getName(), "xLoc", &xLoc);
}

void PointProbe::ioParam_yLoc(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValueRequired(ioFlag, getName(), "yLoc", &yLoc);
}

void PointProbe::ioParam_fLoc(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValueRequired(ioFlag, getName(), "fLoc", &fLoc);
}

void PointProbe::ioParam_batchLoc(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValueRequired(ioFlag, getName(), "batchLoc", &batchLoc);
}

int PointProbe::initNumValues() { return setNumValues(2); }

int PointProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = LayerProbe::communicateInitInfo(message);
   assert(getTargetLayer());
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   bool isRoot           = parent->getCommunicator()->commRank() == 0;
   if ((xLoc < 0 || xLoc > loc->nxGlobal) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: xLoc coordinate %d is out "
            "of bounds (layer has %d neurons in "
            "the x-direction.\n",
            getTargetLayer()->getName(),
            xLoc,
            loc->nxGlobal);
      status = PV_FAILURE;
   }
   if ((yLoc < 0 || yLoc > loc->nyGlobal) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: yLoc coordinate %d is out "
            "of bounds (layer has %d neurons in "
            "the y-direction.\n",
            getTargetLayer()->getName(),
            yLoc,
            loc->nyGlobal);
      status = PV_FAILURE;
   }
   if ((fLoc < 0 || fLoc > loc->nf) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: fLoc coordinate %d is out "
            "of bounds (layer has %d features.\n",
            getTargetLayer()->getName(),
            fLoc,
            loc->nf);
      status = PV_FAILURE;
   }
   if ((batchLoc < 0 || batchLoc > loc->nbatch) && isRoot) {
      ErrorLog().printf(
            "PointProbe on layer %s: batchLoc coordinate %d is "
            "out of bounds (layer has %d "
            "batches.\n",
            getTargetLayer()->getName(),
            batchLoc,
            loc->nbatch);
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS)
      abort();
   return status;
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
int PointProbe::outputState(double timef) {
   getValues(timef);
   return writeState(timef);
}

int PointProbe::calcValues(double timevalue) {
   assert(this->getNumValues() == 2);
   double *valuesBuffer = this->getValuesBuffer();
   // We need to calculate which mpi process contains the target point, and send
   // that info to the root process Each process calculates local index
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   // Calculate local coords from global
   const int kx0         = loc->kx0;
   const int ky0         = loc->ky0;
   const int kb0         = loc->kb0;
   const int nx          = loc->nx;
   const int ny          = loc->ny;
   const int nf          = loc->nf;
   const int nbatch      = loc->nbatch;
   const int xLocLocal   = xLoc - kx0;
   const int yLocLocal   = yLoc - ky0;
   const int nbatchLocal = batchLoc - kb0;

   // if in bounds
   if (xLocLocal >= 0 && xLocLocal < nx && yLocLocal >= 0 && yLocLocal < ny && nbatchLocal >= 0
       && nbatchLocal < nbatch) {
      const float *V        = getTargetLayer()->getV();
      const float *activity = getTargetLayer()->getLayerData();
      // Send V and A to root
      const int k = kIndex(xLocLocal, yLocLocal, fLoc, nx, ny, nf);
      if (V) {
         valuesBuffer[0] = V[k + nbatchLocal * getTargetLayer()->getNumNeurons()];
      }
      else {
         valuesBuffer[0] = 0.0;
      }
      if (activity) {
         const int kex = kIndexExtended(
               k, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         valuesBuffer[1] = activity[kex + nbatchLocal * getTargetLayer()->getNumExtended()];
      }
      else {
         valuesBuffer[1] = 0.0;
      }
      // If not in root process, send to root process
      if (parent->columnId() != 0) {
         MPI_Send(valuesBuffer, 2, MPI_DOUBLE, 0, 0, parent->getCommunicator()->communicator());
      }
   }

   // Root process
   if (parent->columnId() == 0) {
      // Calculate which rank target neuron is
      // TODO we need to calculate rank from batch as well
      int xRank = xLoc / nx;
      int yRank = yLoc / ny;

      int srcRank = rankFromRowAndColumn(
            yRank,
            xRank,
            parent->getCommunicator()->numCommRows(),
            parent->getCommunicator()->numCommColumns());

      // If srcRank is not root process, MPI_Recv from that rank
      if (srcRank != 0) {
         MPI_Recv(
               valuesBuffer,
               2,
               MPI_DOUBLE,
               srcRank,
               0,
               parent->getCommunicator()->communicator(),
               MPI_STATUS_IGNORE);
      }
   }
   return PV_SUCCESS;
}

int PointProbe::writeState(double timevalue) {
   double *valuesBuffer = this->getValuesBuffer();
   if (!mOutputStreams.empty()) {
      output(0).printf("%s t=%.1f V=%6.5f a=%.5f", getMessage(), timevalue, getV(), getA());
      output(0) << std::endl;
   }
   return PV_SUCCESS;
}

double PointProbe::getV() { return getValuesBuffer()[0]; }

double PointProbe::getA() { return getValuesBuffer()[1]; }

} // namespace PV
