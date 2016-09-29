/*
 * Secretary.hpp
 *
 *  Created on Sep 28, 2016
 *      Author: Pete Schultz
 */

#ifndef SECRETARY_HPP_
#define SECRETARY_HPP_

#include "columns/Communicator.hpp"
#include "io/CheckpointEntry.hpp"
#include "io/io.hpp"
#include "io/PVParams.hpp"
#include <map>
#include <memory>

namespace PV {

class Secretary {
public:
   struct TimeInfo {
      double mSimTime = 0.0;
      long int mCurrentCheckpointStep = 0L;
   };
   Secretary(std::string const& name, Communicator * comm);
   ~Secretary();

   void ioParamsFillGroup(enum ParamsIOFlag ioFlag, PVParams * params);
   bool registerCheckpointEntry(std::shared_ptr<CheckpointEntry> checkpointEntry);
   void checkpointRead(std::string const& checkpointReadDir, double * simTimePointer, long int * currentStepPointer);
   void checkpointWrite(std::string const& checkpointWriteDir, double simTime);

   Communicator * getCommunicator() { return mCommunicator; }
   bool doesVerifyWrites() { return mVerifyWritesFlag; }

private:
   std::string mName;
   Communicator * mCommunicator = nullptr;
   std::map<std::string const*, std::shared_ptr<CheckpointEntry> > mCheckpointRegistry;
   TimeInfo mTimeInfo;
   std::shared_ptr<CheckpointEntryData<TimeInfo> > mTimeInfoCheckpointEntry = nullptr; 
   bool mVerifyWritesFlag = true;
};

/**
 * SecretaryDataInterface provides a virtual method intended for interfacing
 * with Secretary register methods.  An object that does checkpointing should
 * derive from SecretaryDataInterface, and override registerData to call
 * Secretary::registerCheckpointEntry once for each piece of checkpointable
 * data.
 *
 * BaseObject derives from SecretaryDataInterface, and calls registerData
 * when it receives a RegisterDataMessage (which HyPerCol::run calls after
 * AllocateDataMessage and before InitializeStateMessage).
 */
class SecretaryDataInterface {
public:
   virtual int registerData(Secretary * secretary, std::string const& objName) { return PV_SUCCESS; }
};

}  // namespace PV

#endif // SECRETARY_HPP_