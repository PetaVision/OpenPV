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
   std::string path(checkpointDirectory);
   int batchWidth = getCommunicator()->numCommBatches();
   if (batchWidth > 1) {
      path.append("/batchsweep_");
      std::size_t lengthLargestBatchIndex = std::to_string(batchWidth - 1).size();
      std::string batchIndexAsString      = std::to_string(getCommunicator()->commBatch());
      std::size_t lengthBatchIndex        = batchIndexAsString.size();
      if (lengthBatchIndex < lengthLargestBatchIndex) {
         path.append(lengthLargestBatchIndex - lengthBatchIndex, '0');
      }
      path.append(batchIndexAsString);
      path.append("/");
   }
   ensureDirExists(getCommunicator(), path.c_str());
   path.append("/").append(getName()).append(".").append(extension);
   return path;
}

void CheckpointEntry::deleteFile(
      std::string const &checkpointDirectory,
      std::string const &extension) const {
   if (getCommunicator()->commRank() == 0) {
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
}
