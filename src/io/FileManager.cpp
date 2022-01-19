#include "FileManager.hpp"
#include "include/pv_common.h"
#include "utils/ExpandLeadingTilde.hpp"

#include <cassert>
#include <cerrno>
#include <cstring>
#include <libgen.h>    // basename and dirname
#include <sys/stat.h>  // mkdir, stat
#include <sys/types.h> // modes used in mkdir
#include <unistd.h>    // sleep

namespace PV {
FileManager::FileManager(
      std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDirectory) {
   mRootProcessRank = 0;
   mMPIBlock = mpiBlock;
   createBlockDirectoryName(baseDirectory);
}

std::shared_ptr<FileManager> FileManager::build(
         std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDirectory) {
   return std::make_shared<FileManager>(mpiBlock, baseDirectory);
}

FileManager::~FileManager() {}

std::shared_ptr<FileStream> FileManager::open(
      std::string const &path,
      std::ios_base::openmode mode,
      bool verifyWrites) const {
    std::shared_ptr<FileStream> stream = nullptr;
    if (isRoot()) {
       std::string modifiedPath = modifyPathForMtoN(path);
       stream = std::make_shared<FileStream>(modifiedPath.c_str(), mode, verifyWrites);
    }
    return stream;   
}

void FileManager::ensureDirectoryExists(std::string const &path) const {
   if (!isRoot()) { return; }
   struct stat statbuffer;
   int statresult = stat(path, statbuffer);

   // if path exists and is a directory, nothing to do.
   // If path exists but is not a directory, fatal error.
   if (statresult == 0) {
      FatalIf(!S_ISDIR(statbuffer.st_mode), "Path \"%s\" exists but is not a directory\n", path.c_str());
      return;
   }

   // Fatal error if checking the path gave an error other than No such file or directory
   FatalIf(
         errno != ENOENT,
         "Error checking status of directory \"%s\": %s.\n",
         path,
         strerror(errno));

   InfoLog().printf(
         "FileManager directory \"%s\" does not exist; attempting to create\n", path.c_str());

   // Try up to mMaxAttempts times until it works
   for (int attemptNum = 0; attemptNum < mMaxAttempts; ++attemptNum) {
      int mkdirstatus = makeDirectory(path);
      if (mkdirstatus != 0 and errno != EEXIST) {
         if (attemptNum == mMaxAttempts - 1) {
            Fatal().printf(
                  "FileManager directory \"%s\" could not be created: %s; Exiting\n",
                  path.c_str(),
                  strerror(errno));
         }
         else {
            getOutputStream().flush();
            WarnLog().printf(
                  "FileManager directory \"%s\" could not be created: %s; Retrying %d out of %d\n",
                  path.c_str(), strerror(errno), attemptNum + 1, mMaxAttempts);
            sleep(1);
         }
      }
      else {
         errno = 0; // It might have been EEXIST but that doesn't count as an error
         // (although perhaps we should verify that it's a directory if it does exist?
         break;
      }
   }
}

void FileManager::changeBaseDirectory(std::string const &newBaseDirectory) {
    createBlockDirectoryName(newBaseDirectory);
}

std::string FileManager::makeBlockFilename(std::string const &path) const {
   return modifyPathForMtoN(path);
}

std::vector<std::string> FileManager::listDirectory() const {
   return listDirectory(".");
}

std::vector<std::string> FileManager::listDirectory(std::string const &path) const {
   std::vector<std::string> result;
   if (!isRoot()) { return result; }
   int status = PV_SUCCESS;
   std::string modifiedPath = modifyPathForMtoN(path);
   DIR *dir = opendir(modifiedPath.c_str());
   if (dir == nullptr) {
      ErrorLog().printf("listDirectory(\"%s\") failed: %s\n", path.c_str(), strerror(errno));
      status = PV_FAILURE;
   }
   if (status == PV_SUCCESS) {
      errno = 0;
      struct dirent *dirEntry;
      for (dirEntry = readdir(dir); dirEntry; dirEntry = readdir(dir)) {
         char const *entryName = dirEntry->d_name;
         if (!std::strcmp(entryName, ".") or !std::strcmp(entryName, "..")) { continue; }
         result.emplace_back(std::string(dirEntry->d_name));
      }
      if (errno) {
         ErrorLog().printf(
               "listDirectory failed to read an entry in \"%s\": %s\n",
               path.c_str(), strerror(errno));
         status = PV_FAILURE;
      }
   }
   FatalIf(
         status != PV_SUCCESS,
         "FileManager failed to list directory \"%s\" in \"%s\"\n",
         path.c_str(),
         mBlockDirectoryName.c_str()); 

   return result;
}

int FileManager::makeDirectory(std::string const &path) const {
   int status     = 0;
   if (!isRoot()) { return status; }

   mode_t dirmode = S_IRWXU | S_IRWXG | S_IRWXO;
   std::string modifiedPath = modifyPathForMtoN(path);
   pvAssert(!modifiedPath.empty());
   std::string::size_type pos = modifiedPath.find('/');
   while (pos != std::string::npos) {
      std::string workingDir = modifiedPath.substr(0, pos);
      status = mkdir(workingDir.c_str(), dirmode);
      if (status != 0 and errno != EEXIST) {
         return status;
      }
      pos = modifiedPath.find('/', pos + 1);
   }
   status = mkdir(modifiedPath.c_str(), dirmode);
   return status;
}

void FileManager::deleteDirectory(std::string const &path) const {
   if (!isRoot()) { return; }
   std::string modifiedPath = modifyPathForMtoN(path);
   for (int attemptNum = 0; attemptNum < mMaxAttempts; ++attemptNum) {
      int rmdirstatus = rmdir(modifiedPath.c_str());
      if (rmdirstatus != 0) {
         if (attemptNum == mMaxAttempts - 1) {
            Fatal().printf(
                  "Directory \"%s\" could not be deleted: %s; Exiting\n",
                  modifiedPath.c_str(), strerror(errno));
         }
         else {
            getOutputStream().flush();
            WarnLog().printf(
                  "Attempt %d of %d: directory \"%s\" could not be deleted: %s\n",
                  attemptNum + 1, mMaxAttempts, modifiedPath.c_str(), strerror(errno));
            sleep(1);
         }
      }
      else {
         if (attemptNum > 0) {
            InfoLog().printf("Attempt %d of %d: directory \"%s\" successfully deleted.\n",
                    attemptNum + 1, mMaxAttempts, modifiedPath.c_str());
            errno = 0;
         }
         return;
      }
   }
}

void FileManager::deleteFile(std::string const &path) const {
   if (!isRoot()) { return; }
   std::string modifiedPath = modifyPathForMtoN(path);
   for (int attemptNum = 0; attemptNum < mMaxAttempts; ++attemptNum) {
      int unlinkstatus = unlink(modifiedPath.c_str());
      if (unlinkstatus != 0) {
         if (attemptNum == mMaxAttempts - 1) {
            Fatal().printf(
                  "File \"%s\" could not be deleted: %s; Exiting\n",
                  modifiedPath.c_str(), strerror(errno));
         }
         else {
            getOutputStream().flush();
            WarnLog().printf("Attempt %d of %d: file \"%s\" could not be deleted: %s\n",
                  attemptNum + 1, mMaxAttempts, modifiedPath.c_str(), strerror(errno));
            sleep(1);
         }
      }
      else {
         if (attemptNum > 0) {
            InfoLog().printf("Attempt %d of %d: file \"%s\" successfully deleted.\n",
                    attemptNum + 1, mMaxAttempts, modifiedPath.c_str());
            errno = 0;
         }
         return;
      }
   }
}

int FileManager::stat(std::string const &path, struct stat &statbuf) const {
   if (!isRoot()) { return 0; }

   std::string modifiedPath = modifyPathForMtoN(path);
   int status = ::stat(modifiedPath.c_str(), &statbuf);
   return status;
}

int FileManager::statRetry(std::string const &path, struct stat &statbuf, int maxAttempts) const {
   if (!isRoot()) { return 0; }

   int status = 0;
   for (int attemptNum = 0; attemptNum < maxAttempts; ++attemptNum) {
      status = stat(path, statbuf);
      if (status) {
         WarnLog().printf(
                 "Attempt %d of %d: unable to get status of \"%s\": %s\n",
                 attemptNum + 1, maxAttempts, path, std::strerror(errno));
      }
      else if (attemptNum > 0) {
         InfoLog().printf("Attempt %d of %d: status of \"%s\" succeeded.\n",
                 attemptNum + 1, maxAttempts, path);
      }
   }
   return status;
}

int FileManager::truncate(std::string const &path, long length) const {
   int status = 0;
   if (isRoot()) {
      std::string modifiedPath = modifyPathForMtoN(path);
      status = ::truncate(modifiedPath.c_str(), static_cast<off_t>(length));
      FatalIf(status, "Unable to truncate \"%s\" to length %ld: %s\n",
            path.c_str(), length, strerror(errno));
   }
   return status;
}

std::string FileManager::modifyPathForMtoN(std::string const &path) const {
   std::string modifiedPath(mBlockDirectoryName);
   assert(!modifiedPath.empty() and modifiedPath.back() == '/');
   modifiedPath.append(path);
   return modifiedPath;
}

void FileManager::createBlockDirectoryName(std::string const &baseDirectory) {
   mBaseDirectory      = baseDirectory;
   mBlockDirectoryName = baseDirectory.empty() ? "." : expandLeadingTilde(baseDirectory);
   assert(!mBlockDirectoryName.empty());

   if (mBlockDirectoryName.back() != '/') {
      mBlockDirectoryName += '/';
   }

   if (mMPIBlock->getGlobalNumRows() != mMPIBlock->getNumRows()
       or mMPIBlock->getGlobalNumColumns() != mMPIBlock->getNumColumns()
       or mMPIBlock->getGlobalBatchDimension() != mMPIBlock->getBatchDimension()) {
      int const blockColumnIndex = mMPIBlock->getStartColumn() / mMPIBlock->getNumColumns();
      int const blockRowIndex    = mMPIBlock->getStartRow() / mMPIBlock->getNumRows();
      int const blockBatchIndex  = mMPIBlock->getStartBatch() / mMPIBlock->getBatchDimension();
      mBlockDirectoryName.append("block_");
      mBlockDirectoryName.append("col" + std::to_string(blockColumnIndex));
      mBlockDirectoryName.append("row" + std::to_string(blockRowIndex));
      mBlockDirectoryName.append("elem" + std::to_string(blockBatchIndex));
      mBlockDirectoryName.append("/");
   }
}

} /* namespace PV */
