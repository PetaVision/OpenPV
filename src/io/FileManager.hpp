/*
 * FileManager.hpp
 *
 *  Created on: April 15, 2021
 *      Author: peteschultz
 */

#ifndef FILEMANAGER_HPP_
#define FILEMANAGER_HPP_

#include "FileStream.hpp"
#include "structures/MPIBlock.hpp"

#include <ios>         // ios_base
#include <memory>      // shared_ptr
#include <string>      // std::string
#include <sys/stat.h>  // stat
#include <vector>      // std::vector

namespace PV {

class FileManager {
  public:
   FileManager(std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDirectory);
   static std::shared_ptr<FileManager> build(
         std::shared_ptr<MPIBlock const> mpiBlock, std::string const &baseDirectory);
   virtual ~FileManager();

   std::shared_ptr<FileStream> open(
         std::string const &path,
         std::ios_base::openmode mode,
         bool verifyWrites = false) const;

   bool isRoot() const { return getRootProcessRank() == mMPIBlock->getRank(); }
   std::vector<std::string> listDirectory() const;
   std::vector<std::string> listDirectory(std::string const &path) const;
   int makeDirectory(std::string const &path) const;
   bool queryFileExists(std::string const &path) const;
   int stat(std::string const &path, struct stat &statbuf) const;
   int statRetry(
         std::string const &path,
         struct stat &statbuf,
         int maxAttempts) const;
   int truncate(std::string const &path, long length) const;

   void deleteDirectory(std::string const &path) const;
   void deleteFile(std::string const &path) const;
   void ensureDirectoryExists(std::string const &path) const;

   /**
    * Changes the directory the FileManager works from.
    * Any previously existing FileStreams created by open() still point to the
    * original location. Any subsequent calls to open(), listDirectory(),
    * etc. will use the new directory.
    * Calling this function does not create the new base directory or
    * check whether the base directory exists or can be created.
    * This function is a work-around for the fact that the outputPath
    * in the params file is not read until after the Communicator creates
    * the OutputFileManager object.
    */
   void changeBaseDirectory(std::string const &newBaseDirectory);

   /**
    * Given a relative path, returns a full path consisting of the effective
    * output directory for the process's checkpoint cell, followed by "/",
    * followed by the given relative path. It is a fatal error for the path to
    * be an absolute path (i.e. starting with '/').
    * This is deprecated, as objects should never need to know the full paths
    * of the files managed by the FileManager object
    */
   std::string makeBlockFilename(std::string const &path) const;

   std::string const &getBaseDirectory() const { return mBaseDirectory; }

   std::shared_ptr<MPIBlock const> getMPIBlock() const { return mMPIBlock; }

   int getRootProcessRank() const { return mRootProcessRank; }

  private:
   std::string modifyPathForMtoN(std::string const &path) const;

   void createBlockDirectoryName(std::string const &baseDirectory);

  private:
   std::string mBaseDirectory;
   std::string mBlockDirectoryName;
   std::shared_ptr<MPIBlock const> mMPIBlock = nullptr;
   int const mMaxAttempts = 5;
   int mRootProcessRank = 0;
};

} /* namespace PV */

#endif // FILEMANAGER_HPP_
