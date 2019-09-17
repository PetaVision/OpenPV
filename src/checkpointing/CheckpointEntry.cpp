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

std::string CheckpointEntry::generatePath(
      std::string const &checkpointDirectory,
      std::string const &extension) const {
   std::string path{checkpointDirectory};
   path.append("/").append(getName());
   if (!extension.empty()) {
      path.append(".").append(extension);
   }
   return path;
}

void CheckpointEntry::deleteFile(
      std::string const &checkpointDirectory,
      std::string const &extension) const {
   if (getMPIBlock()->getRank() == 0) {
      std::string path = generatePath(checkpointDirectory, extension);
      struct stat pathStat;
      int statstatus = stat(path.c_str(), &pathStat);
      if (statstatus == 0) {
         int unlinkstatus = unlink(path.c_str());
         if (unlinkstatus != 0) {
            Fatal().printf("Failure deleting \"%s\": %s\n", path.c_str(), strerror(errno));
         }
      }
   }
}
} // end namespace PV
