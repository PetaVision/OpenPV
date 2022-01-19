/*
 * CheckpointEntry.cpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntry.hpp"
#include "io/fileio.hpp"
#include "utils/PVLog.hpp"
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

namespace PV {

std::string CheckpointEntry::generateFilename(std::string const &extension) const {
   std::string filename = getName();
   if (!extension.empty()) { filename.append(".").append(extension); }
   return filename;
}

std::string CheckpointEntry::generatePath(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &extension) const {
   std::string filename = generateFilename(extension);
   std::string path = fileManager->makeBlockFilename(filename);
   return path;
}

void CheckpointEntry::deleteFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &extension) const {
   std::string filename = generateFilename(extension);
   fileManager->deleteFile(filename);
}

} // end namespace PV
