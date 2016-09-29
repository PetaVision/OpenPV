/*
 * Secretary.cpp
 *
 *  Created on Sep 28, 2016
 *      Author: Pete Schultz
 */

#include "Secretary.hpp"

namespace PV {

Secretary::Secretary(std::string const& name, Communicator * comm) : mName(name), mCommunicator(comm) {}

Secretary::~Secretary() {}

void Secretary::ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams * params) {
   // Currently, HyPerCol reads and writes params related to Secretary functions, and Secretary redundantly reads as needed.
   // TODO: Secretary should read and write from params, and HyPerCol should not need to refer to any Secretary-related params.
   if (ioFlag == PARAMS_IO_READ) {
      params->ioParamValue(ioFlag, mName.c_str(), "verifyWrites", &mVerifyWritesFlag, mVerifyWritesFlag);
   }
   mTimeInfoCheckpointEntry = std::make_shared<CheckpointEntryData<Secretary::TimeInfo> >(
         std::string("timeinfo"), getCommunicator(), &mTimeInfo, (size_t) 1, true/*broadcast*/);
   registerCheckpointEntry(mTimeInfoCheckpointEntry);
}

bool Secretary::registerCheckpointEntry(std::shared_ptr<CheckpointEntry> checkpointEntry) {
   std::string const& name = checkpointEntry->getName();
   bool succeeded = mCheckpointRegistry.insert({&name, checkpointEntry}).second;
   return succeeded;
}

void Secretary::checkpointRead(std::string const& checkpointReadDir, double * simTimePointer, long int * currentStepPointer) {
   for (auto& p : mCheckpointRegistry) {
      double readTime;
      p.second->read(checkpointReadDir, &readTime);
   }
   if (simTimePointer) { *simTimePointer = mTimeInfo.mSimTime; }
   if (currentStepPointer) { *currentStepPointer = mTimeInfo.mCurrentCheckpointStep; }
}

void Secretary::checkpointWrite(std::string const& checkpointWriteDir, double simTime) {
   ensureDirExists(getCommunicator(), checkpointWriteDir.c_str());
   for (auto& p : mCheckpointRegistry) {
      // do timeinfo at the end, so that the presence of timeinfo.bin serves as a flag that the checkpoint has completed
      if (*(p.first)=="timeinfo") { continue; }
      p.second->write(checkpointWriteDir, simTime, mVerifyWritesFlag);
   }

   mTimeInfo.mSimTime = simTime;
   mTimeInfoCheckpointEntry->write(checkpointWriteDir, simTime, mVerifyWritesFlag);
   mTimeInfo.mCurrentCheckpointStep++; // increment at end because checkpoint of initial data is step 0.
}

namespace TextOutput {

template <>
void print(Secretary::TimeInfo const * dataPointer, size_t numValues, std::ostream& stream) {
   for (size_t n=0; n<numValues; n++) {
      stream << "time = " << dataPointer[n].mSimTime << "\n";
      stream << "timestep = " << dataPointer[n].mCurrentCheckpointStep << "\n";
   }
} // print()

}  // namespace TextOutput

}  // namespace PV