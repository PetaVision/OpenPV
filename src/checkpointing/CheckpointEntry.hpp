/*
 * CheckpointEntry.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRY_HPP_
#define CHECKPOINTENTRY_HPP_

#include "io/FileManager.hpp"
#include <memory>
#include <string>

namespace PV {

class CheckpointEntry {
  public:
   CheckpointEntry(std::string const &name)
         : mName(name) {}
   CheckpointEntry(
         std::string const &objName,
         std::string const &dataName) {
      mName = objName;
      if (!(objName.empty() || dataName.empty())) {
         mName.append("_");
      }
      mName.append(dataName);
   }
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag) const {
      return;
   }
   virtual void read(std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
      return;
   }
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const { return; }
   std::string const &getName() const { return mName; }

  protected:
   std::string generateFilename(std::string const &extension) const;

   // deprecated; use generateFilename and FileManager functions.
   std::string
   generatePath(std::shared_ptr<FileManager const> fileManager, std::string const &extension) const;
   void
   deleteFile(std::shared_ptr<FileManager const> fileManager, std::string const &extension) const;

   // data members
  private:
   std::string mName;
};

} // end namespace PV

#endif // CHECKPOINTENTRY_HPP_
