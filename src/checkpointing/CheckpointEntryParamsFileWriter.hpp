/*
 * CheckpointEntryParamsFileWriter.hpp
 *
 *  Created on Jan 16, 2022
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYPARAMSFILEWRITER_HPP_
#define CHECKPOINTENTRYPARAMSFILEWRITER_HPP_

#include "checkpointing/CheckpointEntry.hpp"
#include "io/FileManager.hpp"
#include "observerpattern/Observer.hpp"
#include <memory>

namespace PV {

class CheckpointEntryParamsFileWriter : public CheckpointEntry {
  public:
   CheckpointEntryParamsFileWriter(std::string const &name, Observer *paramsFileWriter)
         : CheckpointEntry(name), mParamsFileWriter(paramsFileWriter) {}
   virtual void write(
         std::shared_ptr<FileManager const> fileManager,
         double simTime, 
         bool verifyWritesFlag) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

   // data members
  private:
   Observer *mParamsFileWriter;
};

} // end namespace PV

#endif // CHECKPOINTENTRYPARAMSFILEWRITER_HPP_
