/*
 * CheckpointEntry.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRY_HPP_
#define CHECKPOINTENTRY_HPP_

#include "structures/MPIBlock.hpp"
#include <memory>
#include <string>

namespace PV {

class CheckpointEntry {
  public:
   CheckpointEntry(std::string const &name, std::shared_ptr<MPIBlock const> mpiBlock)
         : mName(name), mMPIBlock(mpiBlock) {}
   CheckpointEntry(
         std::string const &objName,
         std::string const &dataName,
         std::shared_ptr<MPIBlock const> mpiBlock) {
      mName = objName;
      if (!(objName.empty() || dataName.empty())) {
         mName.append("_");
      }
      mName.append(dataName);
      mMPIBlock = mpiBlock;
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
   std::shared_ptr<MPIBlock const> getMPIBlock() const { return mMPIBlock; }

   // data members
  private:
   std::string mName;
   std::shared_ptr<MPIBlock const> mMPIBlock;
};

} // end namespace PV

#endif // CHECKPOINTENTRY_HPP_
