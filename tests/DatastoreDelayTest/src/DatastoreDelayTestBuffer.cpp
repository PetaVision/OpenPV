/*
 * DatastoreDelayTestBuffer.cpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#include "DatastoreDelayTestBuffer.hpp"
#include <columns/ComponentBasedObject.hpp>
#include <components/BasePublisherComponent.hpp>
#include <include/pv_arch.h>

namespace PV {

DatastoreDelayTestBuffer::DatastoreDelayTestBuffer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DatastoreDelayTestBuffer::~DatastoreDelayTestBuffer() {}

void DatastoreDelayTestBuffer::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
   inited = false; // The first call to updateV sets this to true, so that the class knows whether
   // to initialize or not.
}

Response::Status DatastoreDelayTestBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = InternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *parentPublisher = message->mObjectTable->findObject<BasePublisherComponent>(getName());
   FatalIf(
         parentPublisher == nullptr,
         "%s could not find a BasePublisherComponent.\n",
         getDescription_c());
   mPeriod = parentPublisher->getNumDelayLevels();

   return Response::SUCCESS;
}

void DatastoreDelayTestBuffer::updateBufferCPU(double simTime, double deltaTime) {
   updateV(getLayerLoc(), &inited, mBufferData.data(), mPeriod);
}

int DatastoreDelayTestBuffer::updateV(const PVLayerLoc *loc, bool *inited, float *V, int period) {
   if (*inited) {
      for (int b = 0; b < loc->nbatch; b++) {
         float *VBatch = V + b * loc->nx * loc->ny * loc->nf;
         // Rotate values by one row.
         // Move everything down one row; clobbering row 0 in the process
         for (int y = loc->ny - 1; y > 0; y--) {
            for (int x = 0; x < loc->nx; x++) {
               for (int f = 0; f < loc->nf; f++) {
                  float *V1 = &VBatch[kIndex(x, y, f, loc->nx, loc->ny, loc->nf)];
                  (*V1)--;
                  if (*V1 == 0)
                     *V1 = period;
               }
            }
         }
         // Finally, copy period-th row to zero-th row
         for (int x = 0; x < loc->nx; x++) {
            for (int f = 0; f < loc->nf; f++) {
               VBatch[kIndex(x, 0, f, loc->nx, loc->ny, loc->nf)] =
                     VBatch[kIndex(x, period, f, loc->nx, loc->ny, loc->nf)];
            }
         }
      }
   }
   else {
      for (int b = 0; b < loc->nbatch; b++) {
         float *VBatch = V + b * loc->nx * loc->ny * loc->nf;
         if (loc->ny < period) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            if (rank == 0) {
               ErrorLog().printf(
                     "DatastoreDelayTestBuffer: number of rows (%d) must be >= period (%d).  "
                     "Exiting.\n",
                     loc->ny,
                     period);
            }
            abort();
         }
         int base = loc->ky0;
         for (int x = 0; x < loc->nx; x++) {
            for (int f = 0; f < loc->nf; f++) {
               for (int row = 0; row < loc->ny; row++) {
                  VBatch[kIndex(x, row, f, loc->nx, loc->ny, loc->nf)] = (base + row) % period + 1;
               }
            }
         }
         *inited = true;
      }
   }
   return PV_SUCCESS;
}

} // end of namespace PV block
