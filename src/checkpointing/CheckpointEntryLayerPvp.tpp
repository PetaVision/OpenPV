/*
 * CheckpointEntryLayerPvp.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntryLayerPvp class.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include <cstring>
#include <vector>

#include "io/BroadcastLayerFile.hpp"
#include "io/LayerFile.hpp"
#include "io/SparseBroadcastLayerFile.hpp"
#include "io/SparseLayerFile.hpp"

namespace PV {

template <typename T>
CheckpointEntryLayerPvp<T>::CheckpointEntryLayerPvp(
      std::string const &name,
      PVLayerLoc const *layerLoc,
      bool broadcastFlag,
      bool extendedFlag,
      bool sparseFlag)
      : CheckpointEntry(name) {
   initialize(layerLoc, broadcastFlag, extendedFlag, sparseFlag);
}

template <typename T>
CheckpointEntryLayerPvp<T>::CheckpointEntryLayerPvp(
      std::string const &objName,
      std::string const &dataName,
      PVLayerLoc const *layerLoc,
      bool broadcastFlag,
      bool extendedFlag,
      bool sparseFlag)
      : CheckpointEntry(objName, dataName) {
   initialize(layerLoc, broadcastFlag, extendedFlag, sparseFlag);
}

template <typename T>
void CheckpointEntryLayerPvp<T>::initialize(
      PVLayerLoc const *layerLoc, bool broadcastFlag, bool extendedFlag, bool sparseFlag) {
   mLayerLoc = layerLoc;
   mBroadcastFlag = broadcastFlag;
   mExtendedFlag  = extendedFlag;
   mSparseFlag    = sparseFlag;
}

template <typename T>
void CheckpointEntryLayerPvp<T>::write(
      std::shared_ptr<FileManager const> fileManager,
      double simTime,
      bool verifyWritesFlag) const {
   std::string filename = generateFilename(std::string("pvp"));
   if (mBroadcastFlag && mSparseFlag) {
      writeSparseBroadcast(fileManager, filename, simTime, verifyWritesFlag);
   }
   else if (mBroadcastFlag && !mSparseFlag) {
      writeDenseBroadcast(fileManager, filename, simTime, verifyWritesFlag);
   }
   else if (!mBroadcastFlag && mSparseFlag) {
      writeSparseFile(fileManager, filename, simTime, verifyWritesFlag);
   }
   else /*!mBroadcastFlag && !mSparseFlag*/ {
      writeDenseFile(fileManager, filename, simTime, verifyWritesFlag);
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::writeDenseBroadcast(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      double simTime,
      bool verifyWritesFlag) const {
   BroadcastLayerFile layerFile(
         fileManager,
         filename,
         getLayerLoc()->nf,
         getLayerLoc()->nbatch,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         verifyWritesFlag);
   int const numIndices = getNumIndices();
   for (int i = 0; i < numIndices; ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         T *batchElementStart = calcBatchElementStart(b, i);
         layerFile.setDataLocation(batchElementStart, b);
      }
      layerFile.write(simTime);
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::writeDenseFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      double simTime,
      bool verifyWritesFlag) const {
   LayerFile layerFile(
         fileManager,
         filename,
         *getLayerLoc(),
         getExtendedFlag(),
         false /*fileExtendedFlag*/,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         verifyWritesFlag);
   int const numIndices = getNumIndices();
   for (int i = 0; i < numIndices; ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         T *batchElementStart = calcBatchElementStart(b, i);
         layerFile.setDataLocation(batchElementStart, b);
      }
      layerFile.write(simTime);
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::writeSparseBroadcast(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      double simTime,
      bool verifyWritesFlag) const {
   int nf = getLayerLoc()->nf;
   SparseBroadcastLayerFile layerFile(
         fileManager,
         filename,
         nf,
         getLayerLoc()->nbatch,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         verifyWritesFlag);
   std::vector<SparseList<T>> sparseLists(getLayerLoc()->nbatch);
   for (int i = 0; i < getNumIndices(); ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         SparseList<T> &sparseList = sparseLists[b];
         sparseList.reset(1, 1, nf);
         T *batchElementStart = calcBatchElementStart(b, i);
         for (int f = 0; f < nf; ++f) {
            T const &a = batchElementStart[f];
            if (a != T(0)) {
               sparseList.addEntry(f, a);
            }
         }
         layerFile.setListLocation(&sparseList, b);
      }
      layerFile.write(simTime);
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::writeSparseFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      double simTime,
      bool verifyWritesFlag) const {
   SparseLayerFile layerFile(
         fileManager,
         filename,
         *getLayerLoc(),
         getExtendedFlag(),
         false /*fileExtendedFlag*/,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         verifyWritesFlag);
   std::vector<SparseList<T>> sparseLists(getLayerLoc()->nbatch);
   int width  = getLayerLoc()->nx;
   int height = getLayerLoc()->ny;
   if (getExtendedFlag()) {
      width += getLayerLoc()->halo.lt + getLayerLoc()->halo.rt;
      height += getLayerLoc()->halo.dn + getLayerLoc()->halo.up;
   }
   int nf = getLayerLoc()->nf;
   long numNeuronsL = static_cast<long>(width) * static_cast<long>(height) * static_cast<long>(nf);
   int numNeurons = static_cast<int>(numNeuronsL);
   FatalIf(
         static_cast<long>(numNeurons) != numNeuronsL,
         "Layer for checkpoint file \"%s\" has %ld neurons, which is currently too big for the "
         "sparse pvp file format.\n",
         filename, numNeuronsL);
   for (int i = 0; i < getNumIndices(); ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         SparseList<T> &sparseList = sparseLists[b];
         sparseList.reset(width, height, nf);
         T *batchElementStart = calcBatchElementStart(b, i);
         for (int k = 0; k < numNeurons; ++k) {
            T const &a = batchElementStart[k];
            if (a != T(0)) {
               sparseList.addEntry(k, a);
            }
         }
         layerFile.setListLocation(&sparseList, b);
      }
      layerFile.write(simTime);
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   std::string filename = generateFilename(std::string("pvp"));
   std::vector<double> timeStamps(getNumIndices());
   if (mBroadcastFlag && mSparseFlag) {
      readSparseBroadcast(fileManager, filename, timeStamps);
   }
   else if (mBroadcastFlag && !mSparseFlag) {
      readDenseBroadcast(fileManager, filename, timeStamps);
   }
   else if (!mBroadcastFlag && mSparseFlag) {
      readSparseFile(fileManager, filename, timeStamps);
   }
   else /*!mBroadcastFlag && !mSparseFlag*/ {
      readDenseFile(fileManager, filename, timeStamps);
   }
   applyTimestamps(timeStamps);
   if (!timeStamps.empty()) { *simTimePtr = timeStamps[0]; }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::readDenseBroadcast(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      std::vector<double> &timeStamps) const {
   BroadcastLayerFile layerFile(
         fileManager,
         filename,
         getLayerLoc()->nf,
         getLayerLoc()->nbatch,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   int const numIndices = getNumIndices();
   for (int i = 0; i < numIndices; ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         T *batchElementStart = calcBatchElementStart(b, i);
         layerFile.setDataLocation(batchElementStart, b);
      }
      layerFile.read(timeStamps.at(i));
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::readDenseFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      std::vector<double> &timeStamps) const {
   LayerFile layerFile(
         fileManager,
         filename,
         *getLayerLoc(),
         getExtendedFlag(),
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   int const numIndices = getNumIndices();
   for (int i = 0; i < numIndices; ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         T *batchElementStart = calcBatchElementStart(b, i);
         layerFile.setDataLocation(batchElementStart, b);
      }
      layerFile.read(timeStamps.at(i));
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::readSparseBroadcast(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      std::vector<double> &timeStamps) const {
   int nf = getLayerLoc()->nf;
   SparseBroadcastLayerFile layerFile(
         fileManager,
         filename,
         nf,
         getLayerLoc()->nbatch,
         true /*readOnlyFlag*/,
         true /*clobberFlag*/,
         false /*verifyWrites*/);
   // TODO: implement
   abort();
}

template <typename T>
void CheckpointEntryLayerPvp<T>::readSparseFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &filename,
      std::vector<double> &timeStamps) const {
   SparseLayerFile layerFile(
         fileManager,
         filename,
         *getLayerLoc(),
         getExtendedFlag(),
         false /*fileExtendedFlag*/,
         true /*readOnlyFlag*/,
         true /*clobberFlag*/,
         false /*verifyWritesFlag*/);
   std::vector<SparseList<T>> sparseLists(getLayerLoc()->nbatch);
   int width  = getLayerLoc()->nx;
   int height = getLayerLoc()->ny;
   if (getExtendedFlag()) {
      width += getLayerLoc()->halo.lt + getLayerLoc()->halo.rt;
      height += getLayerLoc()->halo.dn + getLayerLoc()->halo.up;
   }
   int nf = getLayerLoc()->nf;
   long numNeurons = static_cast<long>(width) * static_cast<long>(height) * static_cast<long>(nf);
   for (int i = 0; i < getNumIndices(); ++i) {
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         SparseList<T> &sparseList = sparseLists[b];
         layerFile.setListLocation(&sparseList, b);
      }
      layerFile.read(timeStamps.at(i));
      for (int b = 0; b < getLayerLoc()->nbatch; ++b) {
         SparseList<T> &sparseList = sparseLists[b];
         int checkWidth    = sparseList.getWidth();
         int checkHeight   = sparseList.getHeight();
         int checkFeatures = sparseList.getFeatures();
         bool dimsMatch    =
               (checkWidth == width) and (checkHeight == height) and (checkFeatures == nf);
         FatalIf(
               !dimsMatch,
               "Sparse layer file \"%s\" dimensions %d-by-%d-by-%d do not match expected "
               "dimensions %d-by-%d-by-%d\n",
               filename.c_str(), checkWidth, checkHeight, checkFeatures, width, height, nf);
         loadBatchElement(sparseLists[b], b, i);
      }
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::loadBatchElement(
      SparseList<T> const &sparseList, int batchElement, int index) const {
   int width  = getLayerLoc()->nx;
   int height = getLayerLoc()->ny;
   if (getExtendedFlag()) {
      width += getLayerLoc()->halo.lt + getLayerLoc()->halo.rt;
      height += getLayerLoc()->halo.dn + getLayerLoc()->halo.up;
   }
   int nf = getLayerLoc()->nf;
   long numNeurons = static_cast<long>(width) * static_cast<long>(height) * static_cast<long>(nf);
   T *batchElementStart = calcBatchElementStart(batchElement, index);
   for (long n = 0L; n < numNeurons; ++n) {
      batchElementStart[n] = T(0);
   }
   std::vector<struct SparseEntry<T>> const &sparseData = sparseList.getContents();
   for (auto const &e : sparseData) {
      batchElementStart[e.index] = e.value;
   }
}

template <typename T>
void CheckpointEntryLayerPvp<T>::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, "pvp");
}

} // end namespace PV
