/*
 * CheckpointEntryPvpBuffer.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntryPvpBuffer class.
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
// Read, write and remove methods inherited from CheckpointEntryPvp.

template <typename T>
int CheckpointEntryPvpBuffer<T>::getNumFrames() const {
   return this->getMPIBlock()->getBatchDimension() * this->getLayerLoc()->nbatch;
}

template <typename T>
T *CheckpointEntryPvpBuffer<T>::calcBatchElementStart(int frame) const {
   int const localBatchIndex = frame % this->getLayerLoc()->nbatch;
   int const nx              = this->getLayerLoc()->nx + this->getXMargins();
   int const ny              = this->getLayerLoc()->ny + this->getYMargins();
   return &getDataPointer()[localBatchIndex * nx * ny * this->getLayerLoc()->nf];
}

template <typename T>
int CheckpointEntryPvpBuffer<T>::calcMPIBatchIndex(int frame) const {
   return frame / this->getLayerLoc()->nbatch; // Integer division
}
} // end namespace PV
