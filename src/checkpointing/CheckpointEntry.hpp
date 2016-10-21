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
   CheckpointEntry(std::string const &name, Communicator *communicator)
         : mName(name), mCommunicator(communicator) {}
   CheckpointEntry(
         std::string const &objName,
         std::string const &dataName,
         Communicator *communicator) {
      mName = objName;
      if (!(objName.empty() || dataName.empty())) {
         mName.append("_");
      }
      mName.append(dataName);
      mCommunicator = communicator;
   }
   virtual void
   write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag) const {
      return;
   }
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const { return; }
   virtual void remove(std::string const &checkpointDirectory) const { return; }
   std::string const &getName() const { return mName; }

  protected:
   std::string
   generatePath(std::string const &checkpointDirectory, std::string const &extension) const;
   void deleteFile(std::string const &checkpointDirectory, std::string const &extension) const;
   Communicator *getCommunicator() const { return mCommunicator; }

   // data members
  private:
   std::string mName;
   Communicator *mCommunicator;
};

} // end namespace PV

#endif // CHECKPOINTENTRY_HPP_
