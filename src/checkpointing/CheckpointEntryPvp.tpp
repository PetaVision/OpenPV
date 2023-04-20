/*
 * CheckpointEntryPvp.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntryPvp class.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "io/LayerFile.hpp"
#include <cstring>
#include <vector>

namespace PV {

template <typename T>
CheckpointEntryPvp<T>::CheckpointEntryPvp(
      std::string const &name,
      PVLayerLoc const *layerLoc,
      bool extended)
      : CheckpointEntry(name) {
   initialize(layerLoc, extended);
}

template <typename T>
CheckpointEntryPvp<T>::CheckpointEntryPvp(
      std::string const &objName,
      std::string const &dataName,
      PVLayerLoc const *layerLoc,
      bool extended)
      : CheckpointEntry(objName, dataName) {
   initialize(layerLoc, extended);
}

template <typename T>
void CheckpointEntryPvp<T>::initialize(PVLayerLoc const *layerLoc, bool extended) {
   mLayerLoc = layerLoc;
   mExtended = extended;
}

template <typename T>
void CheckpointEntryPvp<T>::write(
      std::shared_ptr<FileManager const> fileManager,
      double simTime,
      bool verifyWritesFlag) const {
   std::string filename = generateFilename(std::string("pvp"));
   LayerFile layerFile(
         fileManager,
         filename,
         *mLayerLoc,
         mExtended,
         false /*fileExtendedFlag*/,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         verifyWritesFlag);
   int const numIndices = getNumIndices();
   for (int i = 0; i < numIndices; ++i) {
      for (int b = 0; b < mLayerLoc->nbatch; ++b) {
         T *batchElementStart = calcBatchElementStart(b, i);
         layerFile.setDataLocation(batchElementStart, b);
      }
      layerFile.write(simTime);
   }
}

template <typename T>
void CheckpointEntryPvp<T>::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   std::string filename = generateFilename(std::string("pvp"));
   LayerFile layerFile(
         fileManager,
         filename,
         *mLayerLoc,
         mExtended,
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   int const numIndices = getNumIndices();
   std::vector<double> timeStamps(numIndices);
   for (int i = 0; i < numIndices; ++i) {
      for (int b = 0; b < mLayerLoc->nbatch; ++b) {
         T *batchElementStart = calcBatchElementStart(b, i);
         layerFile.setDataLocation(batchElementStart, b);
         clearData(batchElementStart, mLayerLoc, mExtended); // backwards compatibility
         // The clearData() statement can probably be removed safely, but hasn't been tested yet.
      }
      layerFile.read(timeStamps.at(i));
   }
   applyTimestamps(timeStamps);
   if (numIndices > 0) { *simTimePtr = timeStamps[0]; }
}

template <typename T>
void CheckpointEntryPvp<T>::clearData(T *dataStart, PVLayerLoc const *loc, bool extended) const {
   int nx = loc->nx + (extended ? loc->halo.lt + loc->halo.rt : 0);
   int ny = loc->ny + (extended ? loc->halo.dn + loc->halo.up : 0);
   long arraySize = static_cast<long>(nx * ny * loc->nf);
   for (long k = 0; k < arraySize; ++k) {
      dataStart[k] = static_cast<T>(0);
   }
}

template <typename T>
void CheckpointEntryPvp<T>::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, "pvp");
}
} // end namespace PV
