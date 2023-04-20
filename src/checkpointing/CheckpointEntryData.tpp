/*
 * CheckpointEntryData.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntry class hierarchy.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "io/FileStream.hpp"
#include "utils/PVLog.hpp"

namespace PV {

template <typename T>
void CheckpointEntryData<T>::write(
      std::shared_ptr<FileManager const> fileManager,
      double simTime,
      bool verifyWritesFlag) const {
   if (fileManager->isRoot()) {
      std::string filename = generateFilename(std::string("bin"));
      auto fileStream = fileManager->open(filename, std::ios_base::out, verifyWritesFlag);
      fileStream->write(mDataPointer, sizeof(T) * (std::size_t)mNumValues);

      filename = generateFilename(std::string("txt"));
      auto txtStream = fileManager->open(filename, std::ios_base::out, verifyWritesFlag);
      TextOutput::print(mDataPointer, mNumValues, *txtStream);
   }
}

template <typename T>
void CheckpointEntryData<T>::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   if (fileManager->isRoot()) {
      std::string filename = generateFilename(std::string("bin"));
      auto fileStream =
            fileManager->open(filename, std::ios_base::in, false /*verifyWritesFlag not needed*/);
      fileStream->read(mDataPointer, sizeof(T) * (std::size_t)mNumValues);
   }
   if (mBroadcastingFlag) {
      // TODO: Pack all MPI_Bcasts into a single broadcast.
      MPI_Bcast(
            mDataPointer,
            mNumValues * sizeof(T),
            MPI_CHAR,
            0,
            fileManager->getMPIBlock()->getComm());
   }
}

template <typename T>
void CheckpointEntryData<T>::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, "bin");
   deleteFile(fileManager, "txt");
}
} // end namespace PV
