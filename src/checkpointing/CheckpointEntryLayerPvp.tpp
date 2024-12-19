/*
 * CheckpointEntryLayerPvp.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntryLayerPvp class.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "io/BroadcastLayerFile.hpp"
#include "io/LayerFile.hpp"
#include <cstring>
#include <vector>

namespace PV {

template <typename T>
CheckpointEntryLayerPvp<T>::CheckpointEntryLayerPvp(
      std::string const &name,
      PVLayerLoc const *layerLoc,
      bool broadcastFlag,
      bool extendedFlag)
      : CheckpointEntry(name) {
   initialize(layerLoc, broadcastFlag, extendedFlag);
}

template <typename T>
CheckpointEntryLayerPvp<T>::CheckpointEntryLayerPvp(
      std::string const &objName,
      std::string const &dataName,
      PVLayerLoc const *layerLoc,
      bool broadcastFlag,
      bool extendedFlag)
      : CheckpointEntry(objName, dataName) {
   initialize(layerLoc, broadcastFlag, extendedFlag);
}

template <typename T>
void CheckpointEntryLayerPvp<T>::initialize(
      PVLayerLoc const *layerLoc, bool broadcastFlag, bool extendedFlag) {
   mLayerLoc = layerLoc;
   mBroadcastFlag = broadcastFlag;
   mExtendedFlag = extendedFlag;
}

template <typename T>
void CheckpointEntryLayerPvp<T>::write(
      std::shared_ptr<FileManager const> fileManager,
      double simTime,
      bool verifyWritesFlag) const {
   std::string filename = generateFilename(std::string("pvp"));
   if (mBroadcastFlag) {
      BroadcastLayerFile layerFile(
            fileManager,
            filename,
            mLayerLoc->nf,
            mLayerLoc->nbatch,
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
   else {
      LayerFile layerFile(
            fileManager,
            filename,
            *mLayerLoc,
            mExtendedFlag,
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
}

template <typename T>
void CheckpointEntryLayerPvp<T>::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   std::string filename = generateFilename(std::string("pvp"));
   int const numIndices = getNumIndices();
   std::vector<double> timeStamps(numIndices);
   if (mBroadcastFlag) {
      BroadcastLayerFile layerFile(
            fileManager,
            filename,
            mLayerLoc->nf,
            mLayerLoc->nbatch,
            true /*readOnlyFlag*/,
            false /*clobberFlag*/,
            false /*verifyWrites*/);
      for (int i = 0; i < numIndices; ++i) {
         for (int b = 0; b < mLayerLoc->nbatch; ++b) {
            T *batchElementStart = calcBatchElementStart(b, i);
            layerFile.setDataLocation(batchElementStart, b);
         }
         layerFile.read(timeStamps.at(i));
      }
   }
   else {
      LayerFile layerFile(
            fileManager,
            filename,
            *mLayerLoc,
            mExtendedFlag,
            false /*fileExtendedFlag*/,
            true /*readOnlyFlag*/,
            false /*clobberFlag*/,
            false /*verifyWrites*/);
      for (int i = 0; i < numIndices; ++i) {
         for (int b = 0; b < mLayerLoc->nbatch; ++b) {
            T *batchElementStart = calcBatchElementStart(b, i);
            layerFile.setDataLocation(batchElementStart, b);
         }
         layerFile.read(timeStamps.at(i));
      }
   }
   applyTimestamps(timeStamps);
   if (numIndices > 0) { *simTimePtr = timeStamps[0]; }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, "pvp");
}
} // end namespace PV
