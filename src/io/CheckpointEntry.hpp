/*
 * CheckpointEntry.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRY_HPP_
#define CHECKPOINTENTRY_HPP_

#include "columns/Communicator.hpp"
#include <string>

namespace PV {

class CheckpointEntry {
   public:
      CheckpointEntry(std::string const& name, bool verifyingWritesFlag, Communicator * communicator) :
            mName(name), mVerifyingWritesFlag(verifyingWritesFlag), mCommunicator(communicator) {}
      CheckpointEntry(std::string const& objName, std::string const& dataName, bool verifyingWritesFlag, Communicator * communicator) {
         std::string key(objName);
         if (!(objName.empty() || dataName.empty())) { key.append("_"); }
         key.append(dataName);
         mVerifyingWritesFlag = verifyingWritesFlag;
         mCommunicator = communicator;
      }
      virtual void write(std::string const& checkpointDirectory, double simTime) const {return;}
      virtual void read(std::string const& checkpointDirectory, double * simTimePtr) const {return;}
      virtual void remove(std::string const& checkpointDirectory) const {return;}
      std::string const& getName() const { return mName; }
   protected:
      std::string generatePath(std::string const& checkpointDirectory, std::string const& extension) const;
      void deleteFile(std::string const& checkpointDirectory, std::string const& extension) const;
      bool isVerifyingWrites() const { return mVerifyingWritesFlag; }
      Communicator * getCommunicator() const { return mCommunicator; }

// data members
private:
   std::string mName;
   bool mVerifyingWritesFlag;
   Communicator * mCommunicator;
};

template <typename T>
class CheckpointEntryData : public CheckpointEntry {
public:
   CheckpointEntryData(std::string const& name, bool verifyingWritesFlag, Communicator * communicator,
         T * dataPtr, size_t numValues, bool broadcastingFlag) :
         CheckpointEntry(name, verifyingWritesFlag, communicator),
         mDataPointer(dataPtr), mNumValues(numValues), mBroadcastingFlag(broadcastingFlag) {}
   CheckpointEntryData(std::string const& objName, std::string const& dataName, bool verifyingWritesFlag, Communicator * communicator,
         T * dataPtr, size_t numValues, bool broadcastingFlag) :
         CheckpointEntry(objName, dataName, verifyingWritesFlag, communicator),
         mDataPointer(dataPtr), mNumValues(numValues), mBroadcastingFlag(broadcastingFlag) {}
   virtual void write(std::string const& checkpointDirectory, double simTime) const override;
   virtual void read(std::string const& checkpointDirectory, double * simTimePtr) const override;
   virtual void remove(std::string const& checkpointDirectory) const override;

private:
   void broadcast();

private:
   T * mDataPointer;
   size_t mNumValues;
   bool mBroadcastingFlag;
};

template <typename T>
class CheckpointEntryPvp : public CheckpointEntry {
public:
   CheckpointEntryPvp(std::string const& name, bool verifyingWritesFlag, Communicator * communicator,
         T * dataPtr, size_t dataSize, int dataType, PVLayerLoc const * layerLoc, bool extended) :
         CheckpointEntry(name, verifyingWritesFlag, communicator),
         mDataPointer(dataPtr), mDataSize(dataSize), mDataType(dataType), mLayerLoc(layerLoc), mExtended(extended) {}
   CheckpointEntryPvp(std::string const& objName, std::string const& dataName, bool verifyingWritesFlag, Communicator * communicator,
         T * dataPtr, size_t dataSize, int dataType, PVLayerLoc const * layerLoc, bool extended) :
         CheckpointEntry(objName, dataName, verifyingWritesFlag, communicator),
         mDataPointer(dataPtr), mDataSize(dataSize), mDataType(dataType), mLayerLoc(layerLoc), mExtended(extended) {}
   virtual void write(std::string const& checkpointDirectory, double simTime) const override;
   virtual void read(std::string const& checkpointDirectory, double * simTimePtr) const override;
   virtual void remove(std::string const& checkpointDirectory) const override;
private:
   T * mDataPointer;
   size_t mDataSize;
   int mDataType;
   PVLayerLoc const * mLayerLoc = nullptr;
   bool mExtended = false;
};

namespace TextOutput {

template <typename T>
void print(T const * dataPointer, size_t numValues, std::ostream& stream) {
   for (size_t n=0; n<numValues; n++) {
      stream << dataPointer[n] << "\n";
   }
} // end print()

} // end namespace TextOutput
    
} // end namespace PV

#include "CheckpointEntry.tpp"

#endif // CHECKPOINTENTRY_HPP_
