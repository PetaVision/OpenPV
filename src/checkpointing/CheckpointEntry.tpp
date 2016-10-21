/*
 * CheckpointEntry.tpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 *  template implementations for CheckpointEntry class hierarchy.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "io/fileio.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

namespace PV {

template <typename T>
void CheckpointEntryData<T>::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   if (getCommunicator()->commRank() == 0) {
      std::string path    = generatePath(checkpointDirectory, "bin");
      PV_Stream *pvstream = PV_fopen(path.c_str(), "w", verifyWritesFlag);
      int numWritten      = PV_fwrite(mDataPointer, sizeof(T), mNumValues, pvstream);
      if (numWritten != mNumValues) {
         pvError() << "CheckpointEntryData::write: unable to write to \"" << path << "\".\n";
      }
      PV_fclose(pvstream);
      path = generatePath(checkpointDirectory, "txt");
      FileStream txtStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
      TextOutput::print(mDataPointer, mNumValues, txtStream);
   }
}

template <typename T>
void CheckpointEntryData<T>::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   if (getCommunicator()->commRank() == 0) {
      std::string path    = generatePath(checkpointDirectory, "bin");
      PV_Stream *pvstream = PV_fopen(
            path.c_str(),
            "r",
            false /*reading doesn't use verifyWrites, but PV_fopen constructor still takes it as an argument.*/);
      int numRead = PV_fread(mDataPointer, sizeof(T), mNumValues, pvstream);
      if (numRead != mNumValues) {
         pvError() << "CheckpointEntryData::read: unable to read from \"" << path << "\".\n";
      }
      PV_fclose(pvstream);
   }
   if (mBroadcastingFlag) {
      MPI_Bcast(
            mDataPointer,
            mNumValues * sizeof(T),
            MPI_CHAR,
            0,
            getCommunicator()
                  ->communicator()); // TODO: Pack all MPI_Bcasts into a single broadcast.
   }
}

template <typename T>
void CheckpointEntryData<T>::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "bin");
   deleteFile(checkpointDirectory, "txt");
}
}  // end namespace PV
