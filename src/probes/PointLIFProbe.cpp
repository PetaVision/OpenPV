/*
 * PointLIFProbe.cpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#include "PointLIFProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../layers/LIF.hpp"
#include <assert.h>
#include <string.h>

#define NUMBER_OF_VALUES 6 // G_E, G_I, G_IB, V, Vth, A
#define CONDUCTANCE_PRINT_FORMAT "%6.3f"

namespace PV {

PointLIFProbe::PointLIFProbe() : PointProbe() {
   initialize_base();
   // Derived classes of PointLIFProbe should use this PointLIFProbe constructor,
   // and call
   // PointLIFProbe::initialize during their initialization.
}

PointLIFProbe::PointLIFProbe(const char *name, HyPerCol *hc) : PointProbe() {
   initialize_base();
   initialize(name, hc);
}

int PointLIFProbe::initialize_base() {
   writeTime = 0.0;
   writeStep = 0.0;
   return PV_SUCCESS;
}

int PointLIFProbe::initialize(const char *name, HyPerCol *hc) {
   int status = PointProbe::initialize(name, hc);
   writeTime  = 0.0;
   return status;
}

int PointLIFProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PointProbe::ioParamsFillGroup(ioFlag);
   ioParam_writeStep(ioFlag);
   return status;
}

void PointLIFProbe::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   writeStep = parent->getDeltaTime(); // Marian, don't change this default behavior
   parent->parameters()->ioParamValue(
         ioFlag, getName(), "writeStep", &writeStep, writeStep, true /*warnIfAbsent*/);
}

void PointLIFProbe::initNumValues() { setNumValues(NUMBER_OF_VALUES); }

void PointLIFProbe::calcValues(double timevalue) {
   // TODO: Reduce duplicated code between PointProbe::calcValues and
   // PointLIFProbe::calcValues.
   assert(this->getNumValues() == NUMBER_OF_VALUES);
   LIF *LIF_layer = dynamic_cast<LIF *>(getTargetLayer());
   assert(LIF_layer != NULL);
   pvconductance_t const *G_E =
         LIF_layer->getConductance(CHANNEL_EXC) + batchLoc * LIF_layer->getNumNeurons();
   pvconductance_t const *G_I =
         LIF_layer->getConductance(CHANNEL_INH) + batchLoc * LIF_layer->getNumNeurons();
   pvconductance_t const *G_IB =
         LIF_layer->getConductance(CHANNEL_INHB) + batchLoc * LIF_layer->getNumNeurons();
   float const *V        = getTargetLayer()->getV();
   float const *Vth      = LIF_layer->getVth();
   float const *activity = getTargetLayer()->getLayerData();
   assert(V && activity && G_E && G_I && G_IB && Vth);
   double *valuesBuffer = this->getValuesBuffer();
   // We need to calculate which mpi process contains the target point, and send
   // that info to the
   // root process
   // Each process calculates local index
   const PVLayerLoc *loc = getTargetLayer()->getLayerLoc();
   // Calculate local cords from global
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
      const int k      = kIndex(xLocLocal, yLocLocal, fLoc, nx, ny, nf);
      const int kbatch = k + nbatchLocal * getTargetLayer()->getNumNeurons();
      valuesBuffer[0]  = G_E[kbatch];
      valuesBuffer[1]  = G_I[kbatch];
      valuesBuffer[2]  = G_IB[kbatch];
      valuesBuffer[3]  = V[kbatch];
      valuesBuffer[4]  = Vth[kbatch];
      const int kex =
            kIndexExtended(k, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      valuesBuffer[5] = activity[kex + nbatchLocal * getTargetLayer()->getNumExtended()];
      // If not in root process, send to root process
      if (parent->columnId() != 0) {
         MPI_Send(
               valuesBuffer,
               NUMBER_OF_VALUES,
               MPI_DOUBLE,
               0,
               0,
               parent->getCommunicator()->communicator());
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
               NUMBER_OF_VALUES,
               MPI_DOUBLE,
               srcRank,
               0,
               parent->getCommunicator()->communicator(),
               MPI_STATUS_IGNORE);
      }
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
            "%s t=%.1f %d"
            "G_E=" CONDUCTANCE_PRINT_FORMAT " G_I=" CONDUCTANCE_PRINT_FORMAT
            " G_IB=" CONDUCTANCE_PRINT_FORMAT " V=" CONDUCTANCE_PRINT_FORMAT
            " Vth=" CONDUCTANCE_PRINT_FORMAT " a=%.1f",
            getMessage(),
            timevalue,
            k,
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
