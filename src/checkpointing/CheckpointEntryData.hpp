/*
 * CheckpointEntryData.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYDATA_HPP_
#define CHECKPOINTENTRYDATA_HPP_

#include "CheckpointEntry.hpp"
#include "io/PrintStream.hpp"
#include <string>

namespace PV {

template <typename T>
class CheckpointEntryData : public CheckpointEntry {
  public:
   CheckpointEntryData(
         std::string const &name,
         T *dataPtr,
         size_t numValues,
         bool broadcastingFlag)
         : CheckpointEntry(name),
           mDataPointer(dataPtr),
           mNumValues(numValues),
           mBroadcastingFlag(broadcastingFlag) {}
   CheckpointEntryData(
         std::string const &objName,
         std::string const &dataName,
         T *dataPtr,
         size_t numValues,
         bool broadcastingFlag)
         : CheckpointEntry(objName, dataName),
           mDataPointer(dataPtr),
           mNumValues(numValues),
           mBroadcastingFlag(broadcastingFlag) {}
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void
         read(std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  private:
   void broadcast();

  private:
   T *mDataPointer;
   size_t mNumValues;
   bool mBroadcastingFlag;
};

namespace TextOutput {

template <typename T>
void print(T const *dataPointer, size_t numValues, PrintStream &stream) {
   for (size_t n = 0; n < numValues; n++) {
      stream << dataPointer[n] << "\n";
   }
} // end print()

} // end namespace TextOutput

} // end namespace PV

#include "CheckpointEntryData.tpp"

#endif // CHECKPOINTENTRYDATA_HPP_
