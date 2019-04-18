/*
 * CheckpointerDataInterface.hpp
 *
 *  Created on Feb 22, 2018
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTERDATAINTERFACE_HPP_
#define CHECKPOINTERDATAINTERFACE_HPP_

#include "observerpattern/Observer.hpp"

#include "Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"

namespace PV {

/**
 * CheckpointerDataInterface provides a virtual method intended for interfacing
 * with Checkpointer register methods.  An object that does checkpointing should
 * derive from CheckpointerDataInterface and override the following methods:
 *
 * - registerData should call Checkpointer::registerCheckpointEntry
 * once for each piece of data that should be read when restarting a run from
 * a checkpoint. Note that for simple data, where CheckpointEntryData is the
 * appropriate derived class of CheckpointEntry to use, it is convenient to
 * use the Checkpointer::registerCheckpointData method template, which handles
 * creating the shared_ptr needed by registerCheckpointEntry().
 * Note that CheckpointerDataInterface::registerData sets mMPIBlock. Derived
 * classes that override registerData should call
 * CheckpointerDataInterface::registerData in order to use this data member.
 *
 * - readStateFromCheckpoint should call one of the readNamedCheckpointEntry
 * methods for each piece of data that should be read when the object's
 * initializeFromCheckpointFlag is set. The data read by readStateFromCheckpoint
 * must be a subset of the data registered by the registerData function member.
 *
 * BaseObject derives from CheckpointerDataInterface, and calls registerData
 * when it receives a RegisterDataMessage (which HyPerCol::run calls after
 * AllocateDataMessage and before InitializeStateMessage); and calls
 * readStateFromCheckpoint when it receives a ReadStateFromCheckpointMessage
 * (which HyPerCol::run calls after InitializeStateMessage if
 * CheckpointReadDirectory is not set).
 */
class CheckpointerDataInterface : public Observer {
  public:
   virtual Response::Status registerData(Checkpointer *checkpointer);

   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) {
      return Response::NO_ACTION;
   }

   MPIBlock const *getMPIBlock() { return mMPIBlock; }

  protected:
   Response::Status
   respondRegisterData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);
   Response::Status respondReadStateFromCheckpoint(
         std::shared_ptr<ReadStateFromCheckpointMessage<Checkpointer> const> message);

   Response::Status
   respondProcessCheckpointRead(std::shared_ptr<ProcessCheckpointReadMessage const> message);
   Response::Status
   respondPrepareCheckpointWrite(std::shared_ptr<PrepareCheckpointWriteMessage const> message);

   virtual Response::Status processCheckpointRead() { return Response::NO_ACTION; }
   virtual Response::Status prepareCheckpointWrite() { return Response::NO_ACTION; }

  private:
   MPIBlock const *mMPIBlock = nullptr;
};

} // namespace PV

#endif // CHECKPOINTERDATAINTERFACE_HPP_
