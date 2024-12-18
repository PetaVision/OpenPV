/*
 * CheckpointEntryLayerBuffer.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntryLayerBuffer class.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <cstring>
#include <vector>

namespace PV {

// Constructors defined in .hpp file.
// Read, write and remove methods inherited from CheckpointEntryLayerPvp.

template <typename T>
int CheckpointEntryLayerBuffer<T>::getNumIndices() const { return 1; }

template <typename T>
T *CheckpointEntryLayerBuffer<T>::calcBatchElementStart(int batchElement, int index) const {
   PVLayerLoc const *loc = this->getLayerLoc();
   int const nx = loc->nx + (this->getExtended() ? loc->halo.lt + loc->halo.rt : 0);
   int const ny = loc->ny + (this->getExtended() ? loc->halo.dn + loc->halo.up : 0);
   return &getDataPointer()[batchElement * nx * ny * loc->nf];
}

} // end namespace PV
